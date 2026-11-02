"""Matched source-only conditional fitting with epoch-level resumability."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import ContextPredictor, prediction_loss
from .information import evaluate_information
from .data import sha256


PUBLIC_KEYS = ('x', 'neighbors', 'relative_xy', 'neighbor_mask')


def public_batch(batch):
    return {k: batch[k] for k in PUBLIC_KEYS}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = Path(path).with_suffix('.part.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def sample_observations(counts, budgets, generator, identity_only_fraction=.15):
    """Random identities are fixed without inspecting hidden count outcomes."""
    n, genes = counts.shape
    if (not budgets or any(isinstance(k, bool) or int(k) != k or not 0 <= k <= genes for k in budgets)
            or not 0 <= identity_only_fraction <= 1):
        raise ValueError('Legal history budgets and identity-only fraction required')
    lengths = torch.as_tensor(budgets, device=counts.device)
    chosen = lengths[torch.randint(len(lengths), (n,), generator=generator, device=counts.device)]
    ranks = torch.rand((n, genes), device=counts.device, generator=generator).argsort(1)
    observed = torch.zeros_like(counts, dtype=torch.bool)
    observed.scatter_(1, ranks, torch.arange(genes, device=counts.device)[None] < chosen[:, None])
    identity_only = torch.rand(n, device=counts.device, generator=generator) < identity_only_fraction
    value_mask = observed & ~identity_only[:, None]
    values = torch.where(value_mask, counts, torch.zeros_like(counts))
    return values, observed, value_mask


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = ContextPredictor(**payload['model_config'])
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval(), payload


@torch.no_grad()
def monitor_model(model, cohort, normalization, settings, seed, device):
    result = evaluate_information(model, cohort, normalization, seed=seed + 39000,
                                 budgets=[0, 8, 32], batch_size=settings['batch_size'], device=device)
    # Proper-scoring source selection balances image-only and count-conditioned
    # predictions on exactly the same unobserved scopes at each budget.
    ce = [float(result['records'][str(b)][c]['state_ce'].mean())
          for b in (0, 8, 32) for c in ('no_history', 'real_values')]
    cost = [float(result['records'][str(b)]['real_values']['cost'].mean()) for b in (0, 8, 32)]
    return {'state_ce': float(np.mean(ce)), 'remaining_cost': float(np.mean(cost)),
            'criterion': 'source monitor mean 3-class CE; equal budgets and empty/real conditions',
            'count': len(cohort)}


def fit(model, train, monitor, normalization, settings, output, seed, device='cuda'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    complete = output / 'summary.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if sha256(output / 'best.pt') != saved['best_checkpoint_sha256']:
            raise ValueError('Completed best checkpoint changed')
        return load_model(output / 'best.pt', device)[0], saved
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(seed + 75000)
    start_epoch = updates = stale = 0
    records = []
    best_epoch = 0
    model_config = model.get_config()
    last = output / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['settings'] != settings or saved['model_config'] != model_config or saved['seed'] != seed:
            raise ValueError('Resume settings/model/seed changed')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator_state'])
        torch.set_rng_state(saved['torch_rng'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_epoch, updates, stale = saved['epoch'], saved['updates'], saved['stale']
        best, best_epoch, records = saved['best_monitor'], saved['best_epoch'], saved['records']
        print(f'RESUME {output.name} seed{seed} epoch{start_epoch}', flush=True)
    else:
        best = monitor_model(model, monitor, normalization, settings, seed, device)
        records = [{'epoch': 0, 'monitor': best}]
        initial = {'model_config': model_config, 'state_dict': snapshot(model), 'epoch': 0,
                   'seed': seed, 'monitor': best}
        save_checkpoint(output / 'initial.pt', initial)
        save_checkpoint(output / 'best.pt', initial)
    started = time.monotonic()
    finished_early = start_epoch >= settings['min_epochs'] and stale >= settings['patience']
    epoch = start_epoch
    for epoch in range(start_epoch + 1, (start_epoch if finished_early else settings['max_epochs']) + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator, device=device).cpu().numpy()
        losses = []
        epoch_start = time.monotonic()
        for offset in range(0, len(order), settings['batch_size']):
            indices = order[offset:offset + settings['batch_size']]
            batch = train.batch(indices, device, normalization)
            counts = batch['counts']
            values, observed, value_mask = sample_observations(
                counts, settings['history_budgets'], generator, settings['identity_only_fraction'])
            target = ~observed
            public = public_batch(batch)
            conditional = model(public, values, observed, value_mask=value_mask)
            empty_mask = torch.zeros_like(observed)
            empty = model(public, torch.zeros_like(values), empty_mask)
            conditional_loss, _ = prediction_loss(conditional, counts, target, count_weight=settings['count_weight'])
            empty_loss, _ = prediction_loss(empty, counts, target, count_weight=settings['count_weight'])
            loss = .5 * (conditional_loss + empty_loss)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite conditional loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'])
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('Nonfinite gradient norm')
            optimizer.step()
            updates += 1
            losses.append((float(loss.detach()), len(indices)))
        record = {'epoch': epoch, 'loss': sum(v * n for v, n in losses) / len(train),
                  'training_seconds': time.monotonic() - epoch_start}
        checked = epoch % settings['monitor_every'] == 0 or epoch == settings['max_epochs']
        if checked:
            mon = monitor_model(model, monitor, normalization, settings, seed, device)
            record['monitor'] = mon
            if mon['state_ce'] < best['state_ce'] - 1e-8:
                best, best_epoch, stale = mon, epoch, 0
                save_checkpoint(output / 'best.pt', {'model_config': model_config,
                    'state_dict': snapshot(model), 'epoch': epoch, 'seed': seed, 'monitor': mon})
            else:
                stale += 1
            print(f'{output.name} seed{seed} epoch{epoch}: loss={record["loss"]:.5f} '
                  f'monitor_CE={mon["state_ce"]:.5f} best={best_epoch} train_s={record["training_seconds"]:.2f}', flush=True)
        records.append(record)
        save_checkpoint(last, {'model_config': model_config, 'state_dict': snapshot(model),
            'settings': settings, 'seed': seed, 'epoch': epoch, 'updates': updates, 'optimizer': optimizer.state_dict(),
            'generator_state': generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
            'best_epoch': best_epoch, 'best_monitor': best, 'stale': stale, 'records': records})
        write_json(output / 'progress.json', {'best_epoch': best_epoch, 'records': records})
        if checked and epoch >= settings['min_epochs'] and stale >= settings['patience']:
            break
    model, _ = load_model(output / 'best.pt', device)
    summary = {'best_epoch': best_epoch, 'last_epoch': epoch, 'updates': updates, 'records': records,
               'training_cells': len(train), 'monitor_cells': len(monitor), 'best_monitor': best,
               'elapsed_seconds_this_invocation': time.monotonic() - started,
               'settings': settings, 'model_config': model_config, 'seed': seed,
               'test_used_for_selection': False, 'count1_class': 'neutral class1, never a detection negative',
               'best_checkpoint_sha256': sha256(output / 'best.pt')}
    write_json(complete, summary)
    return model, summary
