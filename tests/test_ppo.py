import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from he_gene_rl import core, ppo


def test_initial_policy_reproduces_utility_rule_and_probability_mass():
    torch.manual_seed(3)
    model = ppo.ActorCritic(5, 3, hidden_dim=8)
    features = torch.randn(4, 5)
    utility = torch.tensor([[.5, .2, -.4], [-.5, -.2, -.4], [.1, .8, .3], [1., 2., 3.]])
    selected = torch.tensor([[False, False, False], [False, False, False],
                             [False, True, False], [True, True, True]])
    answer = model.act(features, utility, selected, deterministic=True)
    assert answer['actions'].tolist() == [0, 3, 2, 3]
    # Check the full hierarchical distribution, including an exhausted row.
    output = model(features, utility, selected)
    mass = torch.zeros(4)
    for action in range(4):
        actions = torch.full((4,), action, dtype=torch.long)
        probability = model._statistics(output, actions)['log_prob'].exp()
        valid = ~selected[:, action] if action < 3 else torch.ones(4, dtype=torch.bool)
        mass += torch.where(valid, probability, 0.)
    assert torch.allclose(mass, torch.ones(4), atol=1e-6)
    generator = torch.Generator().manual_seed(17)
    for _ in range(10):
        draw = model.act(features, utility, selected, generator=generator)
        action = draw['actions']
        assert not bool(((action < 3) & selected.gather(1, action.clamp_max(2)[:, None]).squeeze()).any())
        assert torch.isfinite(draw['log_prob']).all()
        assert draw['entropy'][-1] == 0


def make_trace():
    # Two interleaved full episodes: select0(+2),select1(0),STOP;
    # select2(-1),STOP. Labels are attached only after this trace exists.
    actions = torch.tensor([0, 2, 1, 3, 3])
    episodes = torch.tensor([0, 1, 0, 1, 0])
    selected = torch.tensor([[False, False, False], [False, False, False],
                             [True, False, False], [False, False, True], [True, True, False]])
    return {'features': torch.zeros(5, 7), 'base_utility': torch.zeros(5, 3),
            'selected': selected, 'actions': actions, 'old_log_prob': torch.zeros(5),
            'old_value': torch.zeros(5), 'episode_index': episodes,
            'step_index': torch.tensor([0, 0, 1, 1, 2]), 'terminals': actions == 3,
            'final_selected': torch.tensor([[True, True, False], [False, False, True]])}


def test_reward_attachment_is_posthoc_neutral_aware_and_full_return():
    trace = make_trace()
    original = copy.deepcopy(trace)
    counts = np.array([[2, 1, 0], [1, 2, 0]], dtype=np.int64)
    batch, metrics = ppo.attach_rewards(trace, counts, gae_lambda=1.)
    assert batch['rewards'].tolist() == [2., -1., 0., 0., 0.]
    assert batch['returns'].tolist() == [2., -1., 0., 0., 0.]
    assert metrics['cost'] == 1.5
    assert metrics['reward'] == .5
    for key in trace:
        assert torch.equal(original[key], trace[key])
    alternate, _ = ppo.attach_rewards(trace, np.zeros_like(counts), gae_lambda=1.)
    assert not torch.equal(alternate['rewards'], batch['rewards'])
    for key in ('features', 'base_utility', 'selected', 'actions', 'old_log_prob', 'old_value'):
        assert torch.equal(alternate[key], batch[key])
    bad = copy.deepcopy(trace)
    bad['terminals'][2] = True
    with pytest.raises(ValueError, match='STOP'):
        ppo.attach_rewards(bad, counts)


def test_gae_bootstrap_and_terminal_boundaries():
    rewards = torch.tensor([1., 2., -1.])
    values = torch.tensor([.4, .7, .3])
    advantage, returns = ppo.generalized_advantages(rewards, values,
        torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]), torch.tensor([False, True, True]),
        gamma=1., gae_lambda=1.)
    assert torch.allclose(returns, torch.tensor([3., 2., -1.]))
    assert torch.allclose(advantage, returns - values)
    with pytest.raises(ValueError, match='contiguous'):
        ppo.generalized_advantages(rewards, values,
            torch.tensor([0, 0, 1]), torch.tensor([0, 2, 0]), torch.tensor([False, True, True]))


def test_ppo_reward_signal_changes_gene_preferences():
    torch.manual_seed(9)
    model = ppo.ActorCritic(4, 2, hidden_dim=8)
    features, utility = torch.zeros(8, 4), torch.zeros(8, 2)
    actions = torch.tensor([0, 1, 0, 1, 2, 2, 2, 2])
    selected = torch.zeros(8, 2, dtype=torch.bool)
    selected[4:, :] = torch.tensor([[True, False], [False, True], [True, False], [False, True]])
    with torch.no_grad():
        before = model(features[:1], utility[:1], selected[:1])['gene_logits'].clone()
        stats = model.evaluate_actions(features, utility, selected, actions)
    batch = {'features': features, 'base_utility': utility, 'selected': selected, 'actions': actions,
             'old_log_prob': stats['log_prob'], 'old_value': stats['value'],
             'advantages': torch.tensor([2., -1., 2., -1., 0., 0., 0., 0.]),
             'returns': torch.tensor([2., -1., 2., -1., 0., 0., 0., 0.])}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=0)
    result = ppo.ppo_update(model, optimizer, batch,
        {'ppo_epochs': 4, 'minibatch_size': 8, 'target_kl': 10., 'value_weight': .1},
        torch.Generator().manual_seed(8), 'cpu')
    after = model(features[:1], utility[:1], selected[:1])['gene_logits']
    assert result['gradient_updates'] == 4
    assert float((after[0, 0] - after[0, 1]).detach()) > float(before[0, 0] - before[0, 1])
    assert torch.isfinite(after).all()


class ToyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim, self.gene_count = 1, 2
        self.register_buffer('log_mean', torch.zeros(2))
        self.register_buffer('log_scale', torch.ones(2))

    def encode(self, public):
        return {'center': public['x'][:, :1]}

    def score_encoded(self, encoded, values, selected):
        n = len(values)
        utility = values.new_tensor([.5, -.1]).expand(n, -1)
        return {'utility': utility, 'probabilities': values.new_full((n, 2, 3), 1/3),
                'log_count': values.new_full((n, 2), .8)}


class ToyCohort:
    def __init__(self):
        self.features = np.arange(8, dtype=np.float32).reshape(-1, 1) / 8
        self.counts = np.array([[2, 0], [2, 1], [0, 2], [2, 0]] * 2)

    def __len__(self):
        return len(self.features)


def test_exact_iteration_resume_and_frozen_predictor(tmp_path, monkeypatch):
    def public(cohort, indices, normalization, device='cpu'):
        x = torch.tensor(cohort.features[indices], device=device)
        return {'x': x, 'neighbors': x[:, None], 'relative_xy': torch.zeros(len(indices), 1, 2),
                'neighbor_mask': torch.ones(len(indices), 1, dtype=torch.bool)}
    monkeypatch.setattr(core, 'public_from_cohort', public)
    settings = {'max_iterations': 3, 'min_iterations': 3, 'patience': 10,
                'episodes_per_iteration': 8, 'rollout_batch_size': 4,
                'ppo_epochs': 2, 'minibatch_size': 16, 'learning_rate': .002,
                'hidden_dim': 8, 'target_kl': 1.}
    predictor = ToyPredictor()
    cohort = ToyCohort()
    ppo.fit_policy(predictor, cohort, cohort, {}, settings, tmp_path / 'complete', 42, 'cpu')
    save = ppo.save_checkpoint
    def interrupted(path, payload):
        save(path, payload)
        if path.name == 'last.pt' and payload['iteration'] == 1:
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(ppo, 'save_checkpoint', interrupted)
    with pytest.raises(RuntimeError, match='simulated'):
        ppo.fit_policy(predictor, cohort, cohort, {}, settings, tmp_path / 'resumed', 42, 'cpu')
    monkeypatch.setattr(ppo, 'save_checkpoint', save)
    ppo.fit_policy(predictor, cohort, cohort, {}, settings, tmp_path / 'resumed', 42, 'cpu')
    a = torch.load(tmp_path / 'complete/last.pt', weights_only=False)
    b = torch.load(tmp_path / 'resumed/last.pt', weights_only=False)
    for key, value in a['state_dict'].items():
        assert torch.equal(value, b['state_dict'][key]), key
    assert a['updates'] == b['updates']
    assert torch.equal(a['generator_state'], b['generator_state'])
    assert a['best_monitor'] == b['best_monitor']
    assert np.array_equal(a['seen'], b['seen'])
    changed = ToyPredictor()
    changed.log_mean[0] += .01
    with pytest.raises(ValueError, match='predictor'):
        ppo.fit_policy(changed, cohort, cohort, {}, settings, tmp_path / 'complete', 42, 'cpu')
