"""Paired residual-information evaluation on an identical remaining-gene scope.

History identities are chosen using independent randomness or an explicitly
provided public order. Hidden RNA is consulted only to reveal those identities
or to score predictions after model calls. No threshold is fitted here.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F


PUBLIC_KEYS = ('x', 'neighbors', 'relative_xy', 'neighbor_mask')
CONDITIONS = ('no_history', 'identities_only', 'real_values')


def public_batch(batch):
    return {key: batch[key] for key in PUBLIC_KEYS}


def stable_orders(cell_ids, gene_count, seed):
    """Uniform permutations independent of RNA, row ordering and batch size."""
    if gene_count < 1 or int(gene_count) != gene_count:
        raise ValueError('Positive integer gene count required')
    ids = np.asarray(cell_ids).astype(str)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique one-dimensional cell IDs required')
    result = np.empty((len(ids), gene_count), dtype=np.int64)
    for i, cell_id in enumerate(ids):
        digest = hashlib.sha256(f'context-information:{int(seed)}:{cell_id}'.encode()).digest()
        entropy = int.from_bytes(digest[:8], 'little')
        result[i] = np.random.default_rng(entropy).permutation(gene_count)
    return result


def _orders(cohort, seed, order):
    n, genes = cohort.counts.shape
    if order is None:
        return stable_orders(cohort.cell_ids, genes, seed), 'uniform independent cell-ID/seed permutation'
    result = np.asarray(order)
    if (result.shape != (n, genes) or result.dtype.kind not in 'iu'
            or not np.all(np.sort(result, axis=1) == np.arange(genes))):
        raise ValueError('Explicit public order must be a full gene permutation per cell')
    return result.astype(np.int64, copy=False), 'caller-supplied frozen public order; no outcome-based selection'


def _budgets(budgets, genes):
    values = tuple(budgets)
    if not values or any(isinstance(k, bool) or int(k) != k or not 0 <= k <= genes for k in values):
        raise ValueError('Budgets must be integers in [0,gene_count]')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate budgets are not allowed')
    return tuple(sorted(int(k) for k in values))


def _masked_mean(values, mask):
    n = mask.sum(1)
    sums = torch.where(mask, values, torch.zeros_like(values)).sum(1)
    return torch.where(n > 0, sums / n.clamp_min(1), torch.full_like(sums, torch.nan))


@torch.no_grad()
def prediction_metrics(output, counts, target_mask):
    """Per-cell metrics; every condition must use the same target_mask.

raw1 is an explicitly modelled nuisance state for proper utility probabilities.
It is excluded from detection, count regression, FP and FN, as in the task.
"""
    logits, log_count = output['state_logits'], output['log_count']
    if (counts.ndim != 2 or target_mask.shape != counts.shape or target_mask.dtype != torch.bool
            or logits.shape != (*counts.shape, 3) or log_count.shape != counts.shape):
        raise ValueError('Aligned count/mask [B,G], state_logits [B,G,3], log_count [B,G] required')
    if (not bool(torch.isfinite(counts).all()) or bool((counts < 0).any())
            or bool((counts != counts.floor()).any()) or not bool(torch.isfinite(logits).all())
            or not bool(torch.isfinite(log_count).all())):
        raise ValueError('Finite predictions and nonnegative integer counts required')
    labels = torch.where(counts == 0, 0, torch.where(counts == 1, 1, 2)).long()
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    ce = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # Binary probability conditioned on a non-singleton observation. Stable
    # two-logit formulation avoids underflow when the nuisance class dominates.
    binary_logp = F.log_softmax(logits[..., (0, 2)], dim=-1)
    probability = binary_logp[..., 1].exp()
    positive, negative = counts >= 2, counts == 0
    valid = target_mask & (counts != 1)
    bce = -torch.where(positive, binary_logp[..., 1], binary_logp[..., 0])
    utility = 2 * p[..., 2] - p[..., 0]
    selected = target_mask & (utility > 0)
    fp = (selected & negative).sum(1).to(logits.dtype)
    fn = (~selected & target_mask & positive).sum(1).to(logits.dtype)
    return {
        'state_ce': _masked_mean(ce, target_mask),
        'bce': _masked_mean(bce, valid),
        'brier': _masked_mean((probability - positive.to(probability.dtype)).square(), valid),
        'log_mse': _masked_mean((log_count - counts.log1p()).square(), valid),
        'fp': fp, 'fn': fn, 'cost': fp + 2 * fn,
        'selected': selected.sum(1).to(logits.dtype),
        'n_remaining': target_mask.sum(1).to(logits.dtype),
        'n_valid_remaining': valid.sum(1).to(logits.dtype),
    }


@torch.no_grad()
def evaluate_information(model, cohort, normalization, seed, budgets=(0, 4, 8, 16, 32),
                         batch_size=256, device='cuda', order=None):
    """Predict with no history, selected identities, or their real measurements.

The output records contain one vector per metric per condition per budget.
In all three conditions, history genes are excluded from evaluation. These
costs describe remaining-gene predictions, not the cost of a committed policy.
"""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    n, genes = cohort.counts.shape
    if not n:
        raise ValueError('An information evaluation requires nonempty cells')
    budgets = _budgets(budgets, genes)
    orders, order_method = _orders(cohort, seed, order)
    records = {str(k): {name: {} for name in CONDITIONS} for k in budgets}
    model.eval()
    for start in range(0, n, batch_size):
        indices = np.arange(start, min(start + batch_size, n))
        batch = cohort.batch(indices, device, normalization)
        public, counts = public_batch(batch), batch['counts']
        empty = torch.zeros_like(counts, dtype=torch.bool)
        blank = torch.zeros_like(counts)
        encoded = model.encode(public) if hasattr(model, 'encode') and hasattr(model, 'score_encoded') else None

        def score(values, observed, available):
            if encoded is not None:
                return model.score_encoded(encoded, values, observed, value_mask=available)
            return model(public, values, observed, value_mask=available)

        # Compute empty prediction once, but rescore it on each budget's
        # remaining scope. Comparing its all-gene loss with masked loss is wrong.
        no_history = score(blank, empty, empty)
        ranked = torch.as_tensor(orders[indices], device=counts.device, dtype=torch.long)
        for k in budgets:
            observed = torch.zeros_like(empty).scatter_(1, ranked[:, :k], True)
            remaining = ~observed
            values = torch.where(observed, counts, blank)
            if k == 0:
                outputs = {name: no_history for name in CONDITIONS}
            else:
                outputs = {
                    'no_history': no_history,
                    'identities_only': score(blank, observed, empty),
                    'real_values': score(values, observed, observed),
                }
            for name, output in outputs.items():
                for metric, vector in prediction_metrics(output, counts, remaining).items():
                    records[str(k)][name].setdefault(metric, []).append(vector.cpu().numpy())
    combined = {k: {name: {metric: np.concatenate(parts) for metric, parts in metrics.items()}
                    for name, metrics in conditions.items()} for k, conditions in records.items()}
    return {'records': combined, 'orders': orders, 'cell_ids': np.asarray(cohort.cell_ids).astype(str),
            'metadata': {'n_cells': n, 'gene_count': genes, 'budgets': list(budgets), 'seed': int(seed),
                         'history_order': order_method, 'same_remaining_scope_all_conditions': True,
                         'history_truth_used_only_for_revealed_identities': True,
                         'classification': 'raw0 / raw1 / raw>=2; raw1 nuisance class',
                         'detection_metrics': 'raw1 excluded; p_positive=p2/(p0+p2)',
                         'decision': '2*p2-p0 > 0, frozen rule without evaluation-label calibration',
                         'cost_scope': 'remaining genes only; history acquisition costs excluded',
                         'budget_comparison': 'paired conditions within a budget; raw costs across budgets have different target scopes'}}


def paired_block_summary(deltas, tiles, n_bootstrap=2000, seed=42):
    """Spatial-block resampling of paired per-cell values, never gene rows."""
    tiles = np.asarray(tiles)
    if tiles.ndim > 1:
        tiles = np.asarray([repr(tuple(row.tolist())) for row in tiles])
    if tiles.ndim != 1 or not len(tiles) or n_bootstrap < 1:
        raise ValueError('Nonempty spatial blocks and positive bootstrap count required')
    ids, inverse = np.unique(tiles, return_inverse=True)
    draw = np.random.default_rng(seed).integers(len(ids), size=(n_bootstrap, len(ids)))
    result = {'n_cells': len(tiles), 'n_blocks': len(ids), 'bootstrap_replicates': int(n_bootstrap),
              'direction': 'candidate minus reference; lower ce/bce/brier/log_mse/fp/fn/cost is better'}
    for key, value in deltas.items():
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (len(tiles),):
            raise ValueError('Metric vector must align with one spatial identity per cell')
        finite = np.isfinite(vector)
        if not finite.any():
            result[key] = {'mean': None, 'ci95': None, 'n_valid_cells': 0}
            continue
        sizes = np.bincount(inverse, weights=finite, minlength=len(ids))
        sums = np.bincount(inverse, weights=np.where(finite, vector, 0), minlength=len(ids))
        numerators, denominators = sums[draw].sum(1), sizes[draw].sum(1)
        means = numerators[denominators > 0] / denominators[denominators > 0]
        result[key] = {'mean': float(vector[finite].mean()),
                       'ci95': np.quantile(means, [.025, .975]).tolist() if len(ids) > 1 else None,
                       'n_valid_cells': int(finite.sum())}
    return result


def summarize_information(result, tiles, n_bootstrap=2000, seed=42):
    summary = {'metadata': result['metadata'], 'budgets': {}}
    for budget, conditions in result['records'].items():
        means = {condition: {metric: (float(np.asarray(values)[np.isfinite(values)].mean())
                                     if np.isfinite(values).any() else None)
                             for metric, values in records.items()}
                 for condition, records in conditions.items()}
        comparisons = {}
        for reference in ('no_history', 'identities_only'):
            differences = {metric: conditions['real_values'][metric] - conditions[reference][metric]
                           for metric in conditions['real_values']}
            comparisons[f'real_values_vs_{reference}'] = paired_block_summary(
                differences, tiles, n_bootstrap, seed)
        summary['budgets'][budget] = {'means': means, 'paired': comparisons}
    return summary


def average_information(results):
    """Average seeds per cell before spatial bootstrap, retaining paired scope."""
    if not results:
        raise ValueError('At least one result required')
    first = results[0]
    for result in results[1:]:
        if (not np.array_equal(result['cell_ids'], first['cell_ids'])
                or result['records'].keys() != first['records'].keys()):
            raise ValueError('Seed results must have identical cell order and budgets')
    records = {}
    for budget, conditions in first['records'].items():
        records[budget] = {}
        for condition, metrics in conditions.items():
            records[budget][condition] = {}
            for metric in metrics:
                matrix = np.stack([r['records'][budget][condition][metric] for r in results]).astype(np.float64)
                valid = np.isfinite(matrix)
                count = valid.sum(0)
                records[budget][condition][metric] = np.divide(
                    np.where(valid, matrix, 0).sum(0), count,
                    out=np.full(matrix.shape[1], np.nan), where=count > 0)
    return {'records': records, 'cell_ids': first['cell_ids'],
            'orders': first['orders'] if all(np.array_equal(r['orders'], first['orders']) for r in results) else None,
            'metadata': {**first['metadata'], 'n_seeds_averaged': len(results),
                         'seed': [r['metadata'].get('seed') for r in results],
                         'uncertainty': 'seed metrics averaged per cell before paired spatial block bootstrap'}}


def compare_information(candidate, reference, tiles, n_bootstrap=2000, seed=42):
    """Compare matched predictor variants; their histories must be identical."""
    if (not np.array_equal(candidate['cell_ids'], reference['cell_ids'])
            or candidate['records'].keys() != reference['records'].keys()):
        raise ValueError('Predictor comparisons require aligned cell identities and budgets')
    if candidate.get('orders') is not None and reference.get('orders') is not None:
        if not np.array_equal(candidate['orders'], reference['orders']):
            raise ValueError('Predictor comparisons require the same acquired gene identities')
    records = {}
    for budget, conditions in candidate['records'].items():
        records[budget] = {}
        for condition, metrics in conditions.items():
            differences = {metric: value - reference['records'][budget][condition][metric]
                           for metric, value in metrics.items()}
            records[budget][condition] = paired_block_summary(differences, tiles, n_bootstrap, seed)
    return records


@torch.no_grad()
def confidence_orders(model, cohort, normalization, batch_size=256, device='cuda'):
    """Public center-model orders; callers reuse these for both predictor arms."""
    model.eval()
    orders = []
    for start in range(0, len(cohort.counts), batch_size):
        batch = cohort.batch(np.arange(start, min(start + batch_size, len(cohort.counts))), device, normalization)
        counts = batch['counts']
        mask = torch.zeros_like(counts, dtype=torch.bool)
        output = model(public_batch(batch), torch.zeros_like(counts), mask, value_mask=mask)
        p = output['state_logits'].softmax(-1)
        utility = 2 * p[..., 2] - p[..., 0]
        orders.append(utility.argsort(dim=1, descending=True, stable=True).cpu().numpy())
    return np.concatenate(orders)


center_confidence_order = confidence_orders
