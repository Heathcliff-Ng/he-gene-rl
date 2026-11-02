"""Label-free sequential transitions shared by training and deployment.

Only the caller's post-trajectory reward/scoring adapter may access RNA.
Neither predicted_sequence nor collect_episode accepts a count/reward input.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import PUBLIC_KEYS
from .data import public_from_features


def public_from_cohort(cohort, indices, normalization, device='cuda'):
    return public_from_features(cohort, np.asarray(indices), normalization, device)


def state_features(model, encoded, output, values, selected):
    normalized = (values.log1p() - model.log_mean) / model.log_scale
    normalized = torch.where(selected, normalized, torch.zeros_like(normalized))
    return torch.cat((encoded['center'], output['probabilities'].flatten(1),
                      output['log_count'], normalized, selected.to(values.dtype),
                      selected.to(values.dtype).mean(1, keepdim=True)), dim=1)


@torch.no_grad()
def predicted_sequence(model, public, policy=None, *, deterministic=True,
                       generator=None, record=False):
    if set(public) != set(PUBLIC_KEYS):
        raise ValueError('Image-only public inputs required; RNA and reward are forbidden')
    model.eval()
    encoded = model.encode(public)
    n, genes = len(public['x']), model.gene_count
    dtype, device = public['x'].dtype, public['x'].device
    selected = torch.zeros((n, genes), dtype=torch.bool, device=device)
    values = torch.zeros((n, genes), dtype=dtype, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    sequence = torch.full((n, genes), -1, dtype=torch.int64, device=device)
    stop_steps = torch.full((n,), -1, dtype=torch.int64, device=device)
    trace = {}
    for step in range(genes + 1):
        output = model.score_encoded(encoded, values, selected)
        utility = output['utility']
        if not bool(torch.isfinite(utility).all()):
            raise ValueError('Nonfinite prediction')
        rows = torch.nonzero(active, as_tuple=False).flatten()
        if not len(rows):
            break
        if policy is None:
            best, actions = utility[rows].masked_fill(selected[rows], -torch.inf).max(1)
            actions = torch.where(best > 0, actions, torch.full_like(actions, genes))
            if record:
                raise ValueError('Training trace requires a stochastic policy')
        else:
            features = state_features(model, encoded, output, values, selected)[rows]
            decision = policy.act(features, utility[rows], selected[rows],
                                  deterministic=deterministic, generator=generator)
            actions = decision['actions']
            if record:
                current = {'features': features, 'base_utility': utility[rows],
                           'selected': selected[rows].clone(), 'actions': actions,
                           'old_log_prob': decision['log_prob'], 'old_value': decision['value'],
                           'episode_index': rows, 'step_index': torch.full_like(rows, step),
                           'terminals': actions == genes}
                for key, tensor in current.items():
                    trace.setdefault(key, []).append(tensor.detach().clone())
        if bool(((actions < 0) | (actions > genes)).any()):
            raise ValueError('Invalid SELECT/STOP action')
        stopping = actions == genes
        active[rows[stopping]] = False
        stop_steps[rows[stopping]] = step
        selecting_rows = rows[~stopping]
        choosing = actions[~stopping]
        if len(selecting_rows):
            if step == genes or bool(selected[selecting_rows, choosing].any()):
                raise ValueError('Repeated selection or failure to stop at panel exhaustion')
            point_values = output['log_count'][selecting_rows, choosing].expm1()
            if not bool(torch.isfinite(point_values).all()) or bool((point_values < 0).any()):
                raise ValueError('Invalid predicted feedback')
            selected[selecting_rows, choosing] = True
            sequence[selecting_rows, step] = choosing
            values[selecting_rows, choosing] = point_values
    if bool(active.any()) or bool((stop_steps < 0).any()):
        raise AssertionError('Every episode must finish without truncation')
    assert torch.equal(selected.sum(1), stop_steps)
    assert torch.equal(selected.sum(1), (sequence >= 0).sum(1))
    assert bool((values[~selected] == 0).all())
    result = {'selected_mask': selected, 'sequence': sequence, 'predicted_values': values,
              'stop_steps': stop_steps}
    if record:
        result.update({key: torch.cat(tensors, dim=0) for key, tensors in trace.items()})
        result['final_selected'] = selected
    return result


def collect_episode(predictor, policy, cohort, indices, normalization, device='cuda', generator=None):
    public = public_from_cohort(cohort, indices, normalization, device)
    return predicted_sequence(predictor, public, policy, deterministic=False,
                              generator=generator, record=True)


def rollout(model, cohort, normalization, policy=None, batch_size=512, device='cuda'):
    chunks = {}
    if policy is not None:
        policy.eval()
    for start in range(0, len(cohort.features), batch_size):
        indices = np.arange(start, min(start + batch_size, len(cohort.features)))
        public = public_from_cohort(cohort, indices, normalization, device)
        result = predicted_sequence(model, public, policy, deterministic=True)
        for key, tensor in result.items():
            chunks.setdefault(key, []).append(tensor.cpu().numpy())
    return {key: np.concatenate(arrays) for key, arrays in chunks.items()}


def final_metrics(counts, selected):
    counts, selected = np.asarray(counts), np.asarray(selected)
    if counts.shape != selected.shape or selected.dtype != bool:
        raise ValueError('Aligned raw counts and boolean final selections required')
    fp = np.count_nonzero(selected & (counts == 0), axis=1)
    fn = np.count_nonzero(~selected & (counts >= 2), axis=1)
    tp = np.count_nonzero(selected & (counts >= 2), axis=1)
    return {'cell_fp': fp, 'cell_fn': fn, 'cell_tp': tp, 'cell_cost': fp + 2 * fn,
            'cell_selected': selected.sum(1)}
