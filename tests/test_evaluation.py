"""Identity alignment, complete selections and label-ordering boundaries."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from he_gene_rl import evaluation


class EvaluationCohort:
    def __init__(self, spatial_blocks=False):
        self.cell_ids = np.array(['cell_a', 'cell_b'])
        self.gene_ids = np.array(['g0', 'g1', 'g2'])
        self.counts = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
        self.tiles = np.array(['block_a', 'block_b'])
        self.metadata = {'spatial_blocks_available': spatial_blocks}

    def __len__(self):
        return len(self.cell_ids)

    @property
    def gene_count(self):
        return len(self.gene_ids)


def archive(path, cohort, selected, *, cell_ids=None, gene_ids=None, sequence=None):
    values = {'selected_mask': np.asarray(selected, dtype=bool),
              'cell_ids': cohort.cell_ids if cell_ids is None else np.asarray(cell_ids),
              'gene_ids': cohort.gene_ids if gene_ids is None else np.asarray(gene_ids)}
    if sequence is not None:
        values['sequence'] = np.asarray(sequence, dtype=np.int64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return path


def test_scoring_aligns_cell_order_and_keeps_neutral_selections(tmp_path):
    cohort = EvaluationCohort()
    path = archive(tmp_path / 'reordered.npz', cohort,
                   [[True, False, True], [True, True, False]],
                   cell_ids=['cell_b', 'cell_a'],
                   sequence=[[0, 2, -1], [0, 1, -1]])
    result = evaluation.score(path, cohort)
    np.testing.assert_array_equal(result['fp'], [1, 0])
    np.testing.assert_array_equal(result['fn'], [1, 0])
    np.testing.assert_array_equal(result['tp'], [0, 1])
    np.testing.assert_array_equal(result['cost'], [3, 0])
    # Neutral raw1 selections contribute to action count, not FP or TP.
    np.testing.assert_array_equal(result['selected'], [2, 2])


def test_matching_gene_names_in_a_different_order_are_rejected(tmp_path):
    cohort = EvaluationCohort()
    path = archive(tmp_path / 'reordered_genes.npz', cohort,
                   [[True, False, False], [False, True, False]],
                   gene_ids=['g1', 'g0', 'g2'])
    with pytest.raises(ValueError, match='gene panel/order'):
        evaluation.score(path, cohort)


@pytest.mark.parametrize('cell_ids, message', [
    (['cell_a', 'cell_a'], 'Duplicate'),
    (['cell_a', 'unknown_cell'], 'cell sets differ'),
])
def test_prediction_cell_ids_must_be_unique_and_exact(tmp_path, cell_ids, message):
    cohort = EvaluationCohort()
    path = archive(tmp_path / 'wrong_cells.npz', cohort,
                   np.zeros((2, 3), dtype=bool), cell_ids=cell_ids)
    with pytest.raises(ValueError, match=message):
        evaluation.aligned_selection(path, cohort)


@pytest.mark.parametrize('sequence, first_selection, message', [
    ([0, 0, -1], [True, False, False], 'repeats genes'),
    ([0, -1, 2], [True, False, True], 'padding'),
    ([0, 1, -1], [True, False, False], 'differs from final selection'),
    ([0, 3, -1], [True, False, False], 'Invalid padded'),
])
def test_sequence_must_represent_one_complete_irreversible_selection(
        tmp_path, sequence, first_selection, message):
    cohort = EvaluationCohort()
    path = archive(tmp_path / 'invalid_sequence.npz', cohort,
                   [first_selection, [False, False, False]],
                   sequence=[sequence, [-1, -1, -1]])
    with pytest.raises(ValueError, match=message):
        evaluation.aligned_selection(path, cohort)


def test_scoring_requires_labels_but_alignment_does_not(tmp_path):
    cohort = EvaluationCohort()
    cohort.counts = None
    path = archive(tmp_path / 'unlabeled.npz', cohort, np.zeros((2, 3), dtype=bool))
    assert evaluation.aligned_selection(path, cohort).shape == (2, 3)
    with pytest.raises(ValueError, match='held-out counts'):
        evaluation.score(path, cohort)


def metric_vectors(fp, fn):
    fp, fn = np.asarray(fp), np.asarray(fn)
    return {'fp': fp, 'fn': fn, 'cost': fp + 2 * fn}


def test_missing_spatial_blocks_never_produce_a_confidence_interval(monkeypatch):
    cohort = EvaluationCohort(spatial_blocks=False)

    def forbidden_bootstrap(*args, **kwargs):
        raise AssertionError('Missing block identities must not enter bootstrap')

    monkeypatch.setattr(evaluation, 'paired_block_summary', forbidden_bootstrap)
    candidate = metric_vectors([0, 1], [1, 1])
    reference = metric_vectors([1, 2], [1, 2])
    result = evaluation.comparison(candidate, reference, cohort)
    assert result['n_blocks'] is None
    assert result['bootstrap_replicates'] == 0
    assert result['cost']['mean'] == -2.
    for metric in ('fp', 'fn', 'cost'):
        assert result[metric]['ci95'] is None


def test_one_supplied_spatial_block_has_no_resampling_interval():
    cohort = EvaluationCohort(spatial_blocks=True)
    cohort.tiles[:] = 'one_block'
    candidate = metric_vectors([0, 1], [1, 1])
    reference = metric_vectors([1, 2], [1, 2])
    result = evaluation.comparison(candidate, reference, cohort, bootstrap_replicates=20)
    assert result['n_blocks'] == 1
    assert result['cost']['mean'] == -2.
    assert result['cost']['ci95'] is None


def test_full_run_finishes_all_predictions_before_loading_any_truth(tmp_path, monkeypatch):
    cohort = EvaluationCohort()
    config = {'sources': ['Rep1'], 'seeds': [42, 43], 'neighbors': 2,
              'evaluation': {'targets': ['Rep1', 'Rep2'], 'bootstrap_replicates': 20}}
    methods = ('adapted_sequence', 'ppo_sequence')
    expected_predictions = 2 * 2 * len(methods)
    completed = []
    image_loads = []
    heldout_checks = []
    truth_loads = []
    monkeypatch.setattr(evaluation, 'assert_frozen', lambda run: {'config': config})

    def fake_predict(run, manifest, source, target, seed, split, output,
                     device, method, batch_size):
        assert not truth_loads
        archive(output, cohort, [[False, False, True], [True, False, False]],
                sequence=[[2, -1, -1], [0, -1, -1]])
        completed.append((source, target, seed, method))

    def cohort_loader(manifest, target, split, **kwargs):
        if kwargs['require_counts'] is False:
            assert not truth_loads
            image_loads.append(target)
            # The pre-prediction identity check receives no counts attribute.
            return SimpleNamespace(cell_ids=cohort.cell_ids.copy())
        assert kwargs['require_counts'] is True
        assert len(completed) == expected_predictions
        truth_loads.append(target)
        return cohort

    def heldout_identity_check(run, source, target, cell_ids):
        assert not truth_loads
        np.testing.assert_array_equal(cell_ids, cohort.cell_ids)
        heldout_checks.append((source, target))

    monkeypatch.setattr(evaluation, 'predict', fake_predict)
    monkeypatch.setattr(evaluation, 'load_cohort', cohort_loader)
    monkeypatch.setattr(evaluation, 'assert_heldout_cells', heldout_identity_check)
    report = evaluation.evaluate_run(tmp_path / 'run', tmp_path / 'data.json',
                                     tmp_path / 'evaluation', methods=methods)
    assert truth_loads == ['Rep1', 'Rep2']
    assert image_loads == ['Rep1', 'Rep2']
    assert heldout_checks == [('Rep1', 'Rep1'), ('Rep1', 'Rep2')]
    assert report['all_predictions_finished_before_test_labels'] is True
    assert len(report['prediction_sha256']) == expected_predictions
    for direction in report['directions'].values():
        assert direction['n_seeds'] == 2
        assert len(direction['per_seed']) == 4
        comparison = direction['comparisons']['ppo_sequence_vs_adapted_sequence']
        assert comparison['cost']['mean'] == 0.
        assert comparison['cost']['ci95'] is None
