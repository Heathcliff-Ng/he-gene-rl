"""Causal and paired-target checks for residual-information evaluation."""
import numpy as np
import pytest
import torch
from torch import nn

from he_gene_rl.information import (stable_orders, prediction_metrics, evaluate_information,
                         summarize_information, average_information, compare_information,
                         confidence_orders)


class TinyCohort:
    def __init__(self, counts):
        self.counts = np.asarray(counts, np.float32)
        self.cell_ids = np.array([f'cell{i}' for i in range(len(self.counts))])
        self.tiles = np.arange(len(self.counts)) // 2

    def batch(self, indices, device, normalization):
        n = len(indices)
        return {'x': torch.ones(n, 3, device=device),
                'neighbors': torch.zeros(n, 2, 3, device=device),
                'relative_xy': torch.zeros(n, 2, 2, device=device),
                'neighbor_mask': torch.ones(n, 2, dtype=torch.bool, device=device),
                'counts': torch.as_tensor(self.counts[indices], device=device)}


class ToyModel(nn.Module):
    """The acquired gene0 count informs whether gene1 is positive."""
    def forward(self, public, values, observed_mask, value_mask=None):
        assert set(public) == {'x', 'neighbors', 'relative_xy', 'neighbor_mask'}
        assert torch.equal(values[~value_mask], torch.zeros_like(values[~value_mask]))
        n, g = values.shape
        logits = torch.zeros(n, g, 3, device=values.device)
        logits[..., 0] = 2.
        logits[..., 1] = -2.
        # Specific gene ordering is public/model-derived, not count-derived.
        logits[:, -1, 2] = 3.
        known = value_mask[:, 0]
        logits[known, 1, 0] = -2.
        logits[known, 1, 2] = 4.
        return {'state_logits': logits, 'log_count': torch.zeros_like(values)}


def test_uniform_history_orders_are_cell_stable_and_counts_independent():
    ids = np.array(['a', 'b', 'c'])
    first = stable_orders(ids, 12, 42)
    second = stable_orders(ids[[2, 0, 1]], 12, 42)
    assert np.array_equal(first[[2, 0, 1]], second)
    assert np.array_equal(np.sort(first, axis=1), np.tile(np.arange(12), (3, 1)))
    assert not np.array_equal(first, stable_orders(ids, 12, 43))


def test_metrics_ignore_acquired_genes_and_singletons_for_detection():
    truth = torch.tensor([[0., 1., 2., 4.]])
    output = {'state_logits': torch.zeros(1, 4, 3), 'log_count': torch.zeros(1, 4)}
    target = torch.tensor([[False, True, True, False]])
    metrics = prediction_metrics(output, truth, target)
    assert metrics['n_remaining'].item() == 2
    assert metrics['n_valid_remaining'].item() == 1
    assert metrics['fp'].item() == 0 and metrics['fn'].item() == 0
    assert metrics['cost'].item() == 0  # Positive utility selects both target identities.
    assert metrics['brier'].item() == pytest.approx(.25)
    assert metrics['log_mse'].item() == pytest.approx(float(torch.log(torch.tensor(3.)).square()))
    changed = dict(output, state_logits=output['state_logits'].clone(), log_count=output['log_count'].clone())
    changed['state_logits'][:, [0, 3]] = torch.tensor([[-50., 50., 0.], [0., -50., 50.]])
    changed['log_count'][:, [0, 3]] = 999.
    for key, value in prediction_metrics(changed, truth, target).items():
        torch.testing.assert_close(value, metrics[key])


def test_same_remaining_scope_and_identity_only_control():
    cohort = TinyCohort([[0, 2, 1, 0], [0, 2, 1, 0], [0, 2, 0, 1], [0, 2, 0, 1]])
    order = np.tile(np.arange(4), (4, 1))
    result = evaluate_information(ToyModel(), cohort, {}, 42, budgets=(0, 1, 4),
                                 device='cpu', batch_size=2, order=order)
    zero, one, all_seen = (result['records'][str(k)] for k in (0, 1, 4))
    for condition in zero:
        for metric in zero[condition]:
            assert np.array_equal(zero[condition][metric], zero['no_history'][metric])
    for condition in one:
        assert np.array_equal(one[condition]['n_remaining'], np.full(4, 3))
        assert np.array_equal(one[condition]['n_valid_remaining'], one['no_history']['n_valid_remaining'])
    assert (one['real_values']['state_ce'] < one['identities_only']['state_ce']).all()
    assert np.array_equal(one['no_history']['state_ce'], one['identities_only']['state_ce'])
    assert (one['real_values']['cost'] < one['identities_only']['cost']).all()
    assert np.isnan(all_seen['real_values']['state_ce']).all()
    assert (all_seen['real_values']['cost'] == 0).all()
    summary = summarize_information(result, cohort.tiles, n_bootstrap=100)
    delta = summary['budgets']['1']['paired']['real_values_vs_identities_only']
    assert delta['state_ce']['mean'] < 0 and delta['cost']['mean'] == -2


def test_batch_size_does_not_change_history_or_metrics():
    cohort = TinyCohort([[0, 2, 1, 0], [0, 2, 1, 0], [0, 2, 0, 1], [0, 2, 0, 1]])
    a = evaluate_information(ToyModel(), cohort, {}, 42, budgets=(0, 1, 2), device='cpu', batch_size=1)
    b = evaluate_information(ToyModel(), cohort, {}, 42, budgets=(0, 1, 2), device='cpu', batch_size=4)
    assert np.array_equal(a['orders'], b['orders'])
    for budget in a['records']:
        for condition in a['records'][budget]:
            for metric in a['records'][budget][condition]:
                np.testing.assert_allclose(a['records'][budget][condition][metric],
                                           b['records'][budget][condition][metric], equal_nan=True)


def test_confidence_order_does_not_use_truth_and_is_shared_for_model_comparison():
    a = TinyCohort([[0, 2, 1, 0], [0, 2, 1, 0]])
    b = TinyCohort([[7, 0, 0, 9], [3, 0, 0, 2]])
    first = confidence_orders(ToyModel(), a, {}, device='cpu', batch_size=1)
    second = confidence_orders(ToyModel(), b, {}, device='cpu', batch_size=2)
    assert np.array_equal(first, second)
    assert (first[:, 0] == 3).all()


def test_paired_comparison_rejects_different_acquired_genes():
    cohort = TinyCohort([[0, 2, 1, 0], [0, 2, 1, 0]])
    a = evaluate_information(ToyModel(), cohort, {}, 42, budgets=(0, 1), device='cpu')
    b = evaluate_information(ToyModel(), cohort, {}, 43, budgets=(0, 1), device='cpu')
    with pytest.raises(ValueError, match='same acquired'):
        compare_information(a, b, cohort.tiles)


def test_duplicate_seeds_do_not_narrow_spatial_intervals():
    cohort = TinyCohort([[0, 2, 1, 0], [0, 2, 1, 0], [0, 2, 0, 1], [0, 2, 0, 1]])
    one = evaluate_information(ToyModel(), cohort, {}, 42, budgets=(0, 1), device='cpu')
    repeated = average_information([one, one, one])
    a = summarize_information(one, cohort.tiles, n_bootstrap=100)
    b = summarize_information(repeated, cohort.tiles, n_bootstrap=100)
    for budget in a['budgets']:
        for comparison in a['budgets'][budget]['paired']:
            first = a['budgets'][budget]['paired'][comparison]
            second = b['budgets'][budget]['paired'][comparison]
            assert first['n_cells'] == second['n_cells']
            for metric in ('state_ce', 'cost', 'brier', 'log_mse'):
                np.testing.assert_allclose(first[metric]['ci95'], second[metric]['ci95'], atol=1e-7)
                assert first[metric]['mean'] == pytest.approx(second[metric]['mean'], abs=1e-7)


def test_invalid_public_order_and_duplicate_budget_fail():
    cohort = TinyCohort([[0, 2, 1, 0]])
    with pytest.raises(ValueError, match='permutation'):
        evaluate_information(ToyModel(), cohort, {}, 42, budgets=(1,), device='cpu',
                             order=np.zeros((1, 4), dtype=int))
    with pytest.raises(ValueError, match='Duplicate budgets'):
        evaluate_information(ToyModel(), cohort, {}, 42, budgets=(1, 1), device='cpu')
