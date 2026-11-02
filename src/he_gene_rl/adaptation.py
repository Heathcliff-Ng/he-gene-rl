"""Adapt a pretrained predictor to its own predicted-count histories.

The history generator accepts public image tensors only. Expression enters
only the supervised loss after the synthetic prefix has been constructed, or
the final monitor scorer after a complete image-only rollout.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import PUBLIC_KEYS, prediction_loss
from .pretraining import (
    load_model, save_checkpoint, snapshot, write_json,
)
from .data import sha256


def public_from_cohort(cohort, indices, normalization, device):
    from .core import public_from_cohort as assemble
    return assemble(cohort, indices, normalization, device)


@torch.no_grad()
def predicted_prefix(model, public, target_steps):
    """Run each cell to its independently requested prefix or earlier STOP.

The returned history contains continuous expm1(log-count predictions), never
measured counts. A stopped cell cannot resume even if later batch steps would
change its scores. Target lengths affect only when generation is suspended.
"""
    if set(public) != set(PUBLIC_KEYS):
        raise ValueError('Prefix generator accepts only public image/position inputs')
    n, genes = len(public['x']), model.gene_count
    device, dtype = public['x'].device, public['x'].dtype
    target_steps = torch.as_tensor(target_steps, device=device)
    if (target_steps.shape != (n,) or target_steps.dtype not in (torch.int32, torch.int64)
            or bool((target_steps < 0).any()) or bool((target_steps > genes).any())):
        raise ValueError('One integer prefix length in [0,gene_count] per cell required')
    was_training = model.training
    model.eval()
    try:
        encoded = model.encode(public)
        values = torch.zeros((n, genes), dtype=dtype, device=device)
        selected = torch.zeros((n, genes), dtype=torch.bool, device=device)
        active = target_steps > 0
        stopped = torch.zeros(n, dtype=torch.bool, device=device)
        steps = torch.zeros(n, dtype=torch.long, device=device)
        for step in range(int(target_steps.max()) if n else 0):
            if not bool(active.any()):
                break
            output = model.score_encoded(encoded, values, selected)
            utility, log_count = output['utility'], output['log_count']
            if (utility.shape != selected.shape or log_count.shape != selected.shape
                    or not bool(torch.isfinite(utility).all())
                    or not bool(torch.isfinite(log_count).all()) or bool((log_count < 0).any())):
                raise ValueError('Finite utility and nonnegative log-count predictions required')
            best, action = utility.masked_fill(selected, -torch.inf).max(1)
            stop_now = active & (best <= 0)
            stopped |= stop_now
            active &= ~stop_now
            rows = torch.nonzero(active, as_tuple=False).flatten()
            if not len(rows):
                break
            genes_selected = action[rows]
            point = log_count[rows, genes_selected].expm1()
            if not bool(torch.isfinite(point).all()):
                raise ValueError('Nonfinite synthetic count point value')
            assert not bool(selected[rows, genes_selected].any())
            selected[rows, genes_selected] = True
            values[rows, genes_selected] = point
            steps[rows] += 1
            active &= steps < target_steps
        assert torch.equal(steps, selected.sum(1))
        assert bool((steps <= target_steps).all())
        assert bool((values[~selected] == 0).all())
        return {'values': values, 'selected_mask': selected, 'steps': steps,
                'stopped': stopped, 'target_steps': target_steps}
    finally:
        model.train(was_training)


@torch.no_grad()
def monitor_adapter(model, cohort, normalization, settings, device):
    from .core import rollout
    predictions = rollout(model, cohort, normalization, policy=None,
                          batch_size=settings.get('monitor_batch_size', 512), device=device)
    selected = predictions['selected_mask']
    if torch.is_tensor(selected):
        selected = selected.detach().cpu().numpy()
    selected = np.asarray(selected, dtype=bool)
    # The complete prediction is finished before the scorer accesses RNA.
    counts = np.asarray(cohort.counts)
    if selected.shape != counts.shape:
        raise ValueError('Monitor prediction and labels must align')
    fp = (selected & (counts == 0)).sum(1)
    fn = (~selected & (counts >= 2)).sum(1)
    return {'cost': float((fp + 2 * fn).mean()), 'fp': float(fp.mean()),
            'fn': float(fn.mean()), 'selected': float(selected.sum(1).mean()),
            'n_cells': len(selected),
            'criterion': 'source monitor full predicted-history rollout FP+2FN; no true-history monitor'}


def _validate_settings(settings, genes):
    budgets = settings['history_budgets']
    if (not budgets or any(isinstance(k, bool) or int(k) != k or not 0 <= k <= genes for k in budgets)
            or settings['batch_size'] < 1 or settings['max_epochs'] < 1
            or settings['min_epochs'] < 0 or settings.get('monitor_every', 1) < 1
            or settings['patience'] < 1 or settings['learning_rate'] <= 0):
        raise ValueError('Invalid adaptation settings')


def fit_adapter(initial_checkpoint, train, monitor, normalization, settings, output, seed, device='cuda'):
    """Fit on source TRAIN; retain the best source-monitor sequential cost.

Completed outputs are reused after identity checks. Every epoch checkpoint
contains optimizer and random state so interruption after saving last.pt can
resume identically. The supplied teacher checkpoint is never changed.
"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    initial_checkpoint = Path(initial_checkpoint).resolve()
    initial_sha = sha256(initial_checkpoint)
    if set(np.asarray(train.cell_ids).astype(str)) & set(np.asarray(monitor.cell_ids).astype(str)):
        raise ValueError('Source train and monitor cells must be disjoint')
    complete = output / 'summary.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if (saved['settings'] != settings or saved['seed'] != seed
                or saved['initial_checkpoint_sha256'] != initial_sha
                or sha256(output / 'best.pt') != saved['best_checkpoint_sha256']):
            raise ValueError('Completed adaptation configuration or checkpoint changed')
        return load_model(output / 'best.pt', device)[0], saved
    torch.manual_seed(seed + 18000)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed + 18000)
    model, _ = load_model(initial_checkpoint, device)
    _validate_settings(settings, model.gene_count)
    model_config = model.get_config()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings.get('weight_decay', 1e-4))
    generator = torch.Generator(device=device).manual_seed(seed + 18100)
    last = output / 'last.pt'
    start_epoch = updates = stale = best_epoch = 0
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if (saved['settings'] != settings or saved['model_config'] != model_config
                or saved['seed'] != seed or saved['initial_checkpoint_sha256'] != initial_sha):
            raise ValueError('Adaptation resume settings or teacher changed')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator_state'])
        torch.set_rng_state(saved['torch_rng'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_epoch, updates, stale = saved['epoch'], saved['updates'], saved['stale']
        best, best_epoch, records = saved['best_monitor'], saved['best_epoch'], saved['records']
        print(f'RESUME adapter seed{seed} epoch{start_epoch}', flush=True)
    else:
        best = monitor_adapter(model, monitor, normalization, settings, device)
        records = [{'epoch': 0, 'monitor': best}]
        initial = {'model_config': model_config, 'state_dict': snapshot(model), 'epoch': 0,
                   'seed': seed, 'monitor': best, 'initial_checkpoint_sha256': initial_sha}
        save_checkpoint(output / 'initial.pt', initial)
        save_checkpoint(output / 'best.pt', initial)
    started = time.monotonic()
    finished_early = start_epoch >= settings['min_epochs'] and stale >= settings['patience']
    epoch = start_epoch
    budgets = torch.as_tensor(settings['history_budgets'], device=device, dtype=torch.long)
    for epoch in range(start_epoch + 1, (start_epoch if finished_early else settings['max_epochs']) + 1):
        epoch_started = time.monotonic()
        order = torch.randperm(len(train), generator=generator, device=device).cpu().numpy()
        losses, history_steps, stopped_cells = [], [], 0
        for offset in range(0, len(order), settings['batch_size']):
            indices = order[offset:offset + settings['batch_size']]
            public = public_from_cohort(train, indices, normalization, device)
            lengths = budgets[torch.randint(len(budgets), (len(indices),), generator=generator, device=device)]
            history = predicted_prefix(model, public, lengths)
            values, selected = history['values'], history['selected_mask']
            target = ~selected
            model.train()
            conditional = model(public, values, selected)
            empty = model(public, torch.zeros_like(values), torch.zeros_like(selected))
            # RNA is first accessed here, exclusively as supervision.
            counts = torch.as_tensor(np.asarray(train.counts[indices], dtype=np.float32), device=device)
            conditional_loss, _ = prediction_loss(conditional, counts, target,
                                                   count_weight=settings.get('count_weight', .1))
            empty_loss, _ = prediction_loss(empty, counts, target,
                                             count_weight=settings.get('count_weight', .1))
            loss = .5 * (conditional_loss + empty_loss)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite prediction-history adaptation loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get('gradient_clip', 5.))
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('Nonfinite adaptation gradient')
            optimizer.step()
            updates += 1
            losses.append((float(loss.detach()), len(indices)))
            history_steps.append(history['steps'].detach().cpu().numpy())
            stopped_cells += int(history['stopped'].sum())
        record = {'epoch': epoch, 'loss': sum(loss * n for loss, n in losses) / len(train),
                  'mean_generated_history_steps': float(np.concatenate(history_steps).mean()),
                  'fraction_stopped_before_requested_prefix': stopped_cells / len(train),
                  'training_seconds': time.monotonic() - epoch_started}
        checked = epoch % settings.get('monitor_every', 1) == 0 or epoch == settings['max_epochs']
        if checked:
            mon = monitor_adapter(model, monitor, normalization, settings, device)
            record['monitor'] = mon
            if mon['cost'] < best['cost'] - 1e-8:
                best, best_epoch, stale = mon, epoch, 0
                save_checkpoint(output / 'best.pt', {'model_config': model_config,
                    'state_dict': snapshot(model), 'epoch': epoch, 'seed': seed,
                    'monitor': mon, 'initial_checkpoint_sha256': initial_sha})
            else:
                stale += 1
            print(f'adapter seed{seed} epoch{epoch}: loss={record["loss"]:.5f} '
                  f'monitor_cost={mon["cost"]:.5f} best={best_epoch} '
                  f'train_s={record["training_seconds"]:.2f}', flush=True)
        records.append(record)
        save_checkpoint(last, {'model_config': model_config, 'state_dict': snapshot(model),
            'settings': settings, 'seed': seed, 'epoch': epoch, 'updates': updates,
            'optimizer': optimizer.state_dict(), 'generator_state': generator.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
            'best_monitor': best, 'best_epoch': best_epoch, 'stale': stale,
            'records': records, 'initial_checkpoint_sha256': initial_sha})
        write_json(output / 'progress.json', {'best_epoch': best_epoch, 'records': records})
        if checked and epoch >= settings['min_epochs'] and stale >= settings['patience']:
            break
    if sha256(initial_checkpoint) != initial_sha:
        raise ValueError('Protected pretrained teacher changed')
    model, _ = load_model(output / 'best.pt', device)
    summary = {'best_epoch': best_epoch, 'last_epoch': epoch, 'updates': updates,
               'records': records, 'training_cells': len(train), 'monitor_cells': len(monitor),
               'best_monitor': best, 'settings': settings, 'model_config': model_config,
               'seed': seed, 'initial_checkpoint': str(initial_checkpoint),
               'initial_checkpoint_sha256': initial_sha,
               'best_checkpoint_sha256': sha256(output / 'best.pt'),
               'elapsed_seconds_this_invocation': time.monotonic() - started,
               'history_source': 'self-generated continuous predicted-count point values only',
               'counts_used_only_for_loss_or_final_monitor_scoring': True,
               'source_count_normalization_retained': True,
               'full_sequence_cost_monitor': True, 'test_used_for_selection': False}
    write_json(complete, summary)
    return model, summary
