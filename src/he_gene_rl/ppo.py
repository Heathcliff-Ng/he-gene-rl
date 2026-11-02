"""PPO on predicted-count trajectories; labels are used only after rollout.

The decoder remains frozen. Its image-only state and synthetic history feed
both actor and critic. The reward adapter has no path back into state creation.
"""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import sha256
from .pretraining import save_checkpoint, snapshot, write_json


def predictor_fingerprint(model):
    """Stable hash of decoder tensors, including all count-normalization buffers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    digest.update(repr((model.hidden_dim, model.gene_count)).encode())
    return digest.hexdigest()


class ActorCritic(nn.Module):
    """Factorized STOP/continue gate followed by a gene distribution.

STOP logit is -max(unselected learned margins)/temperature + stop residual.
Deterministic inference first applies this gate, then argmaxes gene margins;
it is intentionally not the joint categorical MAP of all gene/STOP events.
At initialization, this exactly reproduces the decoder's utility>0 rule.
"""
    def __init__(self, feature_dim, gene_count, hidden_dim=128, temperature=.25):
        super().__init__()
        if min(feature_dim, gene_count, hidden_dim) < 1 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Positive dimensions and temperature required')
        self.feature_dim, self.gene_count = int(feature_dim), int(gene_count)
        self.hidden_dim, self.temperature = int(hidden_dim), float(temperature)
        self.trunk = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU())
        self.gene_head = nn.Linear(self.hidden_dim, self.gene_count)
        self.stop_head = nn.Linear(self.hidden_dim, 1)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        for head in (self.gene_head, self.stop_head, self.value_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def get_config(self):
        return {'feature_dim': self.feature_dim, 'gene_count': self.gene_count,
                'hidden_dim': self.hidden_dim, 'temperature': self.temperature}

    def forward(self, features, base_utility, selected):
        n = len(features)
        if (features.shape != (n, self.feature_dim) or base_utility.shape != (n, self.gene_count)
                or selected.shape != base_utility.shape or selected.dtype != torch.bool
                or not bool(torch.isfinite(features).all())
                or not bool(torch.isfinite(base_utility).all())):
            raise ValueError('Finite features/base utility and aligned boolean selection mask required')
        hidden = self.trunk(features)
        margins = base_utility + self.gene_head(hidden)
        exhausted = selected.all(1)
        masked = margins.masked_fill(selected, -torch.inf)
        maximum = torch.where(exhausted, torch.zeros_like(margins[:, 0]), masked.max(1).values)
        # Safe finite dummy distribution for exhausted rows; they must STOP.
        gene_logits = torch.where(exhausted[:, None], torch.zeros_like(masked), masked / self.temperature)
        stop_logit = -maximum / self.temperature + self.stop_head(hidden).squeeze(-1)
        return {'gene_logits': gene_logits, 'stop_logit': stop_logit,
                'value': self.value_head(hidden).squeeze(-1), 'exhausted': exhausted}

    def _statistics(self, output, actions):
        stop = actions == self.gene_count
        gene_logp = F.log_softmax(output['gene_logits'], dim=1)
        chosen = gene_logp.gather(1, actions.clamp_max(self.gene_count - 1)[:, None]).squeeze(1)
        logp = torch.where(stop, F.logsigmoid(output['stop_logit']),
                           F.logsigmoid(-output['stop_logit']) + chosen)
        logp = torch.where(output['exhausted'], torch.zeros_like(logp), logp)
        p_stop = output['stop_logit'].sigmoid()
        binary_entropy = -(p_stop * F.logsigmoid(output['stop_logit'])
                           + (1 - p_stop) * F.logsigmoid(-output['stop_logit']))
        # Mask impossible genes before multiplying 0 by -inf.
        categorical_entropy = -(gene_logp.exp() * torch.where(torch.isfinite(gene_logp),
                                  gene_logp, torch.zeros_like(gene_logp))).sum(1)
        entropy = binary_entropy + (1 - p_stop) * categorical_entropy
        entropy = torch.where(output['exhausted'], torch.zeros_like(entropy), entropy)
        return {'log_prob': logp, 'value': output['value'], 'entropy': entropy}

    def act(self, features, base_utility, selected, deterministic=False, generator=None):
        output = self(features, base_utility, selected)
        if deterministic:
            stop = (output['stop_logit'] >= 0) | output['exhausted']
            gene = output['gene_logits'].argmax(1)
        else:
            stop = (torch.rand(len(features), device=features.device, generator=generator)
                    < output['stop_logit'].sigmoid()) | output['exhausted']
            gene = torch.multinomial(output['gene_logits'].softmax(1), 1, generator=generator).squeeze(1)
        action = torch.where(stop, torch.full_like(gene, self.gene_count), gene)
        stats = self._statistics(output, action)
        return {'actions': action, **stats}

    def evaluate_actions(self, features, base_utility, selected, actions):
        if actions.shape != (len(features),) or actions.dtype not in (torch.int32, torch.int64):
            raise ValueError('Integer action vector required')
        if bool(((actions < 0) | (actions > self.gene_count)).any()):
            raise ValueError('Invalid action')
        stop = actions == self.gene_count
        repeats = selected.gather(1, actions.clamp_max(self.gene_count - 1)[:, None]).squeeze(1)
        if bool((~stop & repeats).any()):
            raise ValueError('Repeated or exhausted gene action')
        return self._statistics(self(features, base_utility, selected), actions)


def generalized_advantages(rewards, values, episode_index, step_index, terminals,
                           gamma=1., gae_lambda=.95):
    """Finite episodes; STOP is terminal and exhausted panels get STOP too."""
    n = len(rewards)
    if (not n or rewards.ndim != 1 or any(x.shape != rewards.shape for x in
            (values, episode_index, step_index, terminals)) or terminals.dtype != torch.bool
            or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1
            or not bool(torch.isfinite(rewards).all()) or not bool(torch.isfinite(values).all())):
        raise ValueError('Aligned finite nonempty transition vectors required')
    if bool((episode_index < 0).any()) or bool((step_index < 0).any()):
        raise ValueError('Negative episode/step index')
    _, ids = torch.unique(episode_index, sorted=True, return_inverse=True)
    cells, steps = int(ids.max()) + 1, int(step_index.max()) + 1
    if len(torch.unique(step_index * cells + ids)) != n:
        raise ValueError('Duplicate episode-step')
    valid = torch.zeros((steps, cells), device=rewards.device, dtype=torch.bool)
    terminal_grid = torch.zeros_like(valid)
    reward_grid, value_grid = rewards.new_zeros((steps, cells)), values.detach().new_zeros((steps, cells))
    valid[step_index, ids], terminal_grid[step_index, ids] = True, terminals
    reward_grid[step_index, ids], value_grid[step_index, ids] = rewards, values.detach()
    expected_terminal = valid & ~torch.cat((valid[1:], torch.zeros_like(valid[:1])), dim=0)
    if (not bool(valid[0].all()) or bool((valid[1:] & ~valid[:-1]).any())
            or not torch.equal(terminal_grid, expected_terminal)):
        raise ValueError('Episodes must be contiguous and terminal exactly at their last transition')
    advantage = torch.zeros_like(reward_grid)
    carry = torch.zeros_like(reward_grid[0])
    for t in range(steps - 1, -1, -1):
        continuing = valid[t] & ~terminal_grid[t]
        following = value_grid[t + 1] if t + 1 < steps else torch.zeros_like(carry)
        delta = reward_grid[t] + gamma * following * continuing - value_grid[t]
        carry = torch.where(valid[t], delta + gamma * gae_lambda * carry * continuing, 0.)
        advantage[t] = carry
    flat = advantage[step_index, ids]
    return flat, flat + values.detach()


def attach_rewards(trace, counts, gae_lambda=.95):
    """The only training function accepting labels; never changes trace states."""
    actions, episodes = trace['actions'], trace['episode_index']
    y = torch.as_tensor(counts, device=actions.device, dtype=torch.float32)
    genes = trace['base_utility'].shape[1]
    if (y.ndim != 2 or y.shape[1] != genes or not bool(torch.isfinite(y).all())
            or bool((y < 0).any()) or bool((y != y.floor()).any())
            or bool((episodes >= len(y)).any())):
        raise ValueError('Aligned raw integer counts required for post-trajectory scoring')
    select = actions < genes
    if bool(((actions < 0) | (actions > genes)).any()):
        raise ValueError('Invalid action')
    rewards = torch.zeros(len(actions), device=actions.device, dtype=torch.float32)
    outcomes = y[episodes[select], actions[select]]
    rewards[select] = 2 * (outcomes >= 2).float() - (outcomes == 0).float()
    expected_terminal = actions == genes
    if not torch.equal(trace['terminals'], expected_terminal):
        raise ValueError('Only explicit STOP is terminal; do not truncate trajectories')
    selected = torch.as_tensor(trace['final_selected'], device=y.device, dtype=torch.bool)
    if selected.shape != y.shape:
        raise ValueError('Final committed selections must align with reward outcomes')
    actual = torch.zeros_like(selected, dtype=torch.int64)
    actual.index_put_((episodes[select], actions[select]), torch.ones_like(actions[select]), accumulate=True)
    if bool((actual > 1).any()) or not torch.equal(actual.bool(), selected):
        raise ValueError('Trace repeated a gene or differs from final committed set')
    summed = torch.zeros(len(y), device=y.device).scatter_add_(0, episodes, rewards)
    tp, fp = (selected & (y >= 2)).sum(1), (selected & (y == 0)).sum(1)
    fn = (~selected & (y >= 2)).sum(1)
    if not torch.equal(summed, (2 * tp - fp).float()):
        raise ValueError('Trajectory reward does not equal full-set 2TP-FP')
    advantages, returns = generalized_advantages(rewards, trace['old_value'], episodes,
        trace['step_index'], trace['terminals'], gamma=1., gae_lambda=gae_lambda)
    batch = {k: trace[k].detach() for k in ('features', 'base_utility', 'selected', 'actions',
             'old_log_prob', 'old_value', 'episode_index', 'step_index', 'terminals')}
    batch.update(rewards=rewards.detach(), advantages=advantages.detach(), returns=returns.detach())
    metrics = {'episodes': len(y), 'transitions': len(actions), 'reward': float(summed.mean()),
               'fp': float(fp.float().mean()), 'fn': float(fn.float().mean()),
               'cost': float((fp + 2 * fn).float().mean()), 'selected': float(selected.sum(1).float().mean())}
    return batch, metrics


def ppo_update(policy, optimizer, batch, settings, generator, device):
    """Clipped PPO actor objective and bootstrapped GAE critic regression."""
    n = len(batch['actions'])
    advantages = batch['advantages'].float()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    stats, updates, stop = [], 0, False
    policy.train()
    for epoch in range(int(settings.get('ppo_epochs', 4))):
        order = torch.randperm(n, device=device, generator=generator).cpu()
        for start in range(0, n, int(settings.get('minibatch_size', 2048))):
            ids = order[start:start + int(settings.get('minibatch_size', 2048))]
            features, utility, selected, actions, old_logp, target = [batch[key][ids].to(device) for key in
                ('features', 'base_utility', 'selected', 'actions', 'old_log_prob', 'returns')]
            current = policy.evaluate_actions(features, utility, selected, actions)
            advantage = advantages[ids].to(device)
            log_ratio = current['log_prob'] - old_logp
            ratio = log_ratio.exp()
            clip = float(settings.get('clip_ratio', .2))
            policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
            value_loss = .5 * (current['value'] - target).square().mean()
            entropy = current['entropy'].mean()
            loss = (policy_loss + float(settings.get('value_weight', .5)) * value_loss
                    - float(settings.get('entropy_weight', .002)) * entropy)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite PPO objective')
            approximate_kl = ((ratio - 1) - log_ratio).mean()
            if float(approximate_kl.detach()) > float(settings.get('target_kl', .03)):
                stop = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_gradient = torch.stack([p.grad.detach().square().sum()
                for head in (policy.gene_head, policy.stop_head) for p in head.parameters()
                if p.grad is not None]).sum().sqrt()
            gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings.get('gradient_clip', 1.)))
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError('Nonfinite PPO gradient')
            optimizer.step()
            updates += 1
            stats.append({'policy_loss': float(policy_loss.detach()), 'value_loss': float(value_loss.detach()),
                          'entropy': float(entropy.detach()), 'approximate_kl': float(approximate_kl.detach()),
                          'clip_fraction': float(((ratio - 1).abs() > clip).float().mean().detach()),
                          'gradient_norm': float(gradient.detach()),
                          'actor_gradient_norm': float(actor_gradient)})
        if stop:
            break
    policy.eval()
    return {**({key: float(np.mean([r[key] for r in stats])) for key in stats[0]} if stats else {}),
            'gradient_updates': updates, 'kl_early_stop': stop, 'transitions': n}


def load_policy(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = ActorCritic(**payload['model_config'])
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval(), payload


@torch.no_grad()
def monitor_policy(predictor, policy, cohort, normalization, device, batch_size=512):
    from .core import rollout
    records = rollout(predictor, cohort, normalization, policy=policy, batch_size=batch_size, device=device)
    selected = np.asarray(records['selected_mask'], dtype=bool)
    counts = cohort.counts
    fp, fn = (selected & (counts == 0)).sum(1), (~selected & (counts >= 2)).sum(1)
    return {'cost': float((fp + 2 * fn).mean()), 'fp': float(fp.mean()), 'fn': float(fn.mean()),
            'selected': float(selected.sum(1).mean()), 'n_cells': len(cohort),
            'criterion': 'source monitor full-set FP+2FN; predicted histories and deterministic policy only'}


def fit_policy(predictor, train, monitor, normalization, settings, output, seed, device='cuda'):
    from .core import collect_episode
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(settings)
    decoder_sha256 = predictor_fingerprint(predictor)
    model_config = {'feature_dim': predictor.hidden_dim + 6 * predictor.gene_count + 1,
                    'gene_count': predictor.gene_count, 'hidden_dim': settings.get('hidden_dim', 128),
                    'temperature': settings.get('temperature', .25)}
    if (output / 'summary.json').exists():
        summary = json.loads((output / 'summary.json').read_text())
        if (summary['settings'] != settings or summary['seed'] != seed
                or summary['model_config'] != model_config
                or summary['predictor_sha256'] != decoder_sha256):
            raise ValueError('Completed PPO settings/seed/config/frozen predictor mismatch')
        if sha256(output / 'best.pt') != summary['best_checkpoint_sha256']:
            raise ValueError('Completed policy checkpoint changed')
        return load_policy(output / 'best.pt', device)[0], summary
    predictor.eval().requires_grad_(False)
    torch.manual_seed(seed + 62000)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed + 62000)
    generator = torch.Generator(device=device).manual_seed(seed + 63000)
    policy = ActorCritic(**model_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings.get('learning_rate', 1e-4),
                                 weight_decay=settings.get('weight_decay', 0.))
    start, stale, updates, best_iteration, history = 0, 0, 0, 0, []
    seen = np.zeros(len(train), dtype=bool)
    last = output / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if (saved['settings'] != settings or saved['seed'] != seed
                or saved['model_config'] != policy.get_config()
                or saved['predictor_sha256'] != decoder_sha256):
            raise ValueError('Resume settings/config/seed/frozen predictor mismatch')
        policy.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator_state'])
        torch.set_rng_state(saved['torch_rng'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, stale, updates = saved['iteration'], saved['stale'], saved['updates']
        best_iteration, best, history, seen = saved['best_iteration'], saved['best_monitor'], saved['history'], saved['seen']
        print(f'RESUME PPO seed{seed} iteration{start}', flush=True)
    else:
        best = monitor_policy(predictor, policy, monitor, normalization, device,
                              settings.get('monitor_batch_size', 512))
        initial = {'model_config': policy.get_config(), 'state_dict': snapshot(policy),
                   'iteration': 0, 'seed': seed, 'monitor': best,
                   'predictor_sha256': decoder_sha256}
        save_checkpoint(output / 'initial.pt', initial)
        save_checkpoint(output / 'best.pt', initial)
        history = [{'iteration': 0, 'monitor': best}]
    begin = time.monotonic()
    minimum, maximum = int(settings.get('min_iterations', 10)), int(settings.get('max_iterations', 30))
    patience = int(settings.get('patience', 8))
    finished = start >= minimum and stale >= patience
    iteration = start
    for iteration in range(start + 1, (start if finished else maximum) + 1):
        tick = time.monotonic()
        n = min(len(train), int(settings.get('episodes_per_iteration', 1024)))
        chosen = torch.randperm(len(train), device=device, generator=generator)[:n].cpu().numpy()
        seen[chosen] = True
        batches, rollout_metrics = [], []
        for offset in range(0, n, int(settings.get('rollout_batch_size', 128))):
            ids = chosen[offset:offset + int(settings.get('rollout_batch_size', 128))]
            policy.eval()
            # No labels enter this call. Attach them only after complete STOP.
            trace = collect_episode(predictor, policy, train, ids, normalization, device, generator)
            batch, metrics = attach_rewards(trace, train.counts[ids], settings.get('gae_lambda', .95))
            batches.append({k: v.detach().cpu() for k, v in batch.items()})
            rollout_metrics.append(metrics)
            del trace, batch
        combined = {k: torch.cat([part[k] for part in batches]) for k in batches[0]}
        del batches
        optim = ppo_update(policy, optimizer, combined, settings, generator, device)
        updates += optim['gradient_updates']
        record = {'iteration': iteration, 'optimizer': optim, 'episodes': n,
                  'rollout': {k: sum(m[k] * m['episodes'] for m in rollout_metrics) / n
                              for k in ('reward', 'fp', 'fn', 'cost', 'selected')},
                  'seen_training_cells': int(seen.sum()), 'seconds': time.monotonic() - tick}
        del combined
        if iteration % int(settings.get('monitor_every', 1)) == 0 or iteration == maximum:
            mon = monitor_policy(predictor, policy, monitor, normalization, device,
                                 settings.get('monitor_batch_size', 512))
            record['monitor'] = mon
            if mon['cost'] < best['cost'] - 1e-8:
                best, best_iteration, stale = mon, iteration, 0
                save_checkpoint(output / 'best.pt', {'model_config': policy.get_config(),
                    'state_dict': snapshot(policy), 'iteration': iteration, 'seed': seed, 'monitor': mon,
                    'predictor_sha256': decoder_sha256})
            else:
                stale += 1
            print(f'PPO seed{seed} iter{iteration}: monitor_cost={mon["cost"]:.5f} '
                  f'best={best_iteration} updates={updates} rollout_cost={record["rollout"]["cost"]:.5f}', flush=True)
        history.append(record)
        save_checkpoint(last, {'model_config': policy.get_config(), 'state_dict': snapshot(policy),
            'optimizer': optimizer.state_dict(), 'settings': settings, 'seed': seed,
            'iteration': iteration, 'stale': stale, 'updates': updates, 'best_iteration': best_iteration,
            'best_monitor': best, 'history': history, 'seen': seen,
            'predictor_sha256': decoder_sha256,
            'generator_state': generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []})
        write_json(output / 'progress.json', {'best_iteration': best_iteration, 'history': history})
        if iteration >= minimum and stale >= patience:
            break
    initial, _ = load_policy(output / 'initial.pt', device)
    last_changed = any(not torch.equal(value, initial.state_dict()[key]) for key, value in policy.state_dict().items())
    policy, _ = load_policy(output / 'best.pt', device)
    changed = any(not torch.equal(value, initial.state_dict()[key]) for key, value in policy.state_dict().items())
    if predictor_fingerprint(predictor) != decoder_sha256:
        raise ValueError('Frozen decoder tensors changed during PPO')
    summary = {'best_iteration': best_iteration, 'last_iteration': iteration, 'gradient_updates': updates,
               'best_monitor': best, 'initial_monitor': history[0]['monitor'], 'history': history,
               'best_parameters_changed': changed,
               'last_parameters_changed': last_changed, 'predictor_sha256': decoder_sha256,
               'best_is_initial_policy': best_iteration == 0,
               'training_cells': len(train), 'seen_training_cells': int(seen.sum()),
               'monitor_cells': len(monitor), 'test_used_for_selection': False,
               'predictor_frozen': True, 'reward_enters_policy_state': False,
               'settings': settings, 'model_config': policy.get_config(), 'seed': seed,
               'elapsed_seconds_this_invocation': time.monotonic() - begin,
               'best_checkpoint_sha256': sha256(output / 'best.pt')}
    write_json(output / 'summary.json', summary)
    return policy, summary
