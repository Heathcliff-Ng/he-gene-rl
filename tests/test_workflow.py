"""Portable deployment guards; no training is performed by these tests."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from he_gene_rl import io, workflow


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    source = run / 'source'
    directory = source / 'seed42'
    for phase in ('pretraining', 'adaptation', 'ppo'):
        (directory / phase).mkdir(parents=True)
        (directory / phase / 'best.pt').write_bytes(b'checkpoint fixture; no torch loading')
    (source / 'genes.json').write_text(json.dumps(['g0', 'g1']))
    np.savez(source / 'normalization.npz', feature_mean=np.zeros(2, np.float32),
             feature_scale=np.ones(2, np.float32), coordinate_scale=np.array(1.))
    np.savez(source / 'split_ids.npz', train=np.array(['fit-a']), policy=np.array(['fit-b']),
             monitor=np.array(['fit-c']))

    class ImagesOnly:
        features = np.ones((2, 2), np.float32)
        feature_dim = 2
        gene_count = 2
        gene_ids = np.array(['g0', 'g1'])
        cell_ids = np.array(['heldout-a', 'heldout-b'])
        metadata = {'spatial_blocks_available': False}

        def __len__(self):
            return len(self.features)

        @property
        def counts(self):
            raise AssertionError('Deployment touched expression targets')

    cohort = ImagesOnly()
    calls = []

    def loader(manifest, target, split, **kwargs):
        calls.append((manifest, target, split, kwargs))
        assert kwargs['require_counts'] is False
        return cohort

    model = SimpleNamespace(feature_dim=2, gene_count=2)
    actor = object()
    monkeypatch.setattr(workflow, 'runtime', lambda device: None)
    monkeypatch.setattr(workflow, 'assert_frozen', lambda path: {
        'config': {'sources': ['source'], 'seeds': [42], 'neighbors': 8}})
    monkeypatch.setattr(workflow, 'load_cohort', loader)
    monkeypatch.setattr(workflow, 'load_model', lambda path, device: (model, {}))
    monkeypatch.setattr(workflow, 'load_policy', lambda path, device: (
        actor, {'predictor_sha256': 'matching-predictor'}))
    monkeypatch.setattr(workflow, 'predictor_fingerprint', lambda predictor: 'matching-predictor')

    def rollout(predictor, images, normalization, policy, batch_size, device):
        assert images is cohort and predictor is model and policy is actor
        return {'selected_mask': np.array([[True, False], [False, True]]),
                'sequence': np.array([[0, -1], [1, -1]]),
                'predicted_values': np.array([[2., 0.], [0., 3.]], np.float32),
                'stop_steps': np.ones(2, dtype=np.int64)}

    monkeypatch.setattr(workflow, 'rollout', rollout)
    return SimpleNamespace(run=run, cohort=cohort, calls=calls,
                           output=tmp_path / 'prediction.npz')


def test_prediction_loads_only_images_and_keeps_targets_out_of_output(deployment):
    case = deployment
    workflow.predict(case.run, 'portable/manifest.json', 'source', 'target', 42,
                     'predict', case.output, device='cpu')
    assert len(case.calls) == 1
    assert case.calls[0][3]['require_counts'] is False
    with np.load(case.output, allow_pickle=False) as saved:
        assert 'counts' not in saved.files and 'tiles' not in saved.files
        np.testing.assert_array_equal(saved['cell_ids'], case.cohort.cell_ids)
        np.testing.assert_array_equal(saved['gene_ids'], case.cohort.gene_ids)
    sidecar = json.loads(case.output.with_suffix('.npz.json').read_text())
    assert sidecar['test_expression_read'] is False
    assert sidecar['prediction_sha256'] == io.sha256(case.output)


def test_policy_from_different_predictor_is_rejected_before_prediction(deployment, monkeypatch):
    case = deployment
    monkeypatch.setattr(workflow, 'load_policy', lambda path, device: (
        object(), {'predictor_sha256': 'another-predictor'}))
    monkeypatch.setattr(workflow, 'rollout', lambda *args, **kwargs: pytest.fail('Reached rollout'))
    with pytest.raises(ValueError, match='different predictor'):
        workflow.predict(case.run, 'manifest.json', 'source', 'target', 42,
                         'predict', case.output, device='cpu')
    assert not case.output.exists()


def test_prediction_rejects_reordered_genes_before_loading_models(deployment, monkeypatch):
    case = deployment
    case.cohort.gene_ids = np.array(['g1', 'g0'])
    monkeypatch.setattr(workflow, 'load_model', lambda *args: pytest.fail('Loaded model'))
    with pytest.raises(ValueError, match='gene panel/order'):
        workflow.predict(case.run, 'manifest.json', 'source', 'target', 42,
                         'predict', case.output, device='cpu')
    assert not case.output.exists()


def test_same_source_test_rejects_fitted_cell_identity(deployment, monkeypatch):
    case = deployment
    case.cohort.cell_ids = np.array(['fit-b', 'heldout-b'])
    monkeypatch.setattr(workflow, 'load_model', lambda *args: pytest.fail('Loaded model'))
    with pytest.raises(ValueError, match='overlap fitted'):
        workflow.predict(case.run, 'manifest.json', 'source', 'source', 42,
                         'test', case.output, device='cpu')
    assert not case.output.exists()


def test_prediction_preserves_existing_output_before_loading_data(deployment):
    case = deployment
    case.output.write_bytes(b'previous user result')
    with pytest.raises(FileExistsError, match='Preserving'):
        workflow.predict(case.run, 'manifest.json', 'source', 'target', 42,
                         'predict', case.output, device='cpu')
    assert case.output.read_bytes() == b'previous user result'
    assert not case.calls


def test_frozen_run_relocates_and_rejects_modified_artifacts(tmp_path):
    root = tmp_path / 'original'
    directory = root / 'source'
    directory.mkdir(parents=True)
    config = {'sources': ['source'], 'seeds': [42]}
    for name in ('run.json', 'training_summary.json', 'training_inputs.json'):
        (root / name).write_text('{}')
    (root / 'run.json').write_text(json.dumps({'config': config, 'code': io.source_fingerprint()}))
    for name in ('normalization.npz', 'genes.json', 'split_ids.npz'):
        (directory / name).write_bytes(b'identity fixture')
    for stage in ('pretraining', 'adaptation', 'ppo'):
        path = directory / 'seed42' / stage
        path.mkdir(parents=True)
        (path / 'best.pt').write_bytes(stage.encode())
    io.freeze_run(root, config)
    moved = tmp_path / 'moved'
    shutil.copytree(root, moved)
    assert io.assert_frozen(moved)['complete'] is True
    (moved / 'source/seed42/ppo/best.pt').write_bytes(b'replaced model')
    with pytest.raises(ValueError, match='changed frozen artifact'):
        io.assert_frozen(moved)


def test_training_rejects_overlapping_cells_and_gene_order_before_fitting():
    def part(ids, genes=None):
        return SimpleNamespace(cell_ids=np.asarray(ids), gene_ids=np.asarray(
            genes if genes is not None else [f'g{i}' for i in range(32)]),
            feature_dim=4, gene_count=32)

    config = {'pretraining': {'history_budgets': [0, 4, 8, 32]},
              'adaptation': {'history_budgets': [0, 4, 8, 32]}}
    parts = {'train': part(['same']), 'policy': part(['same']), 'monitor': part(['monitor'])}
    with pytest.raises(ValueError, match='Overlapping'):
        workflow._check_training_parts(parts, config)
    parts['policy'] = part(['different'], [f'g{i}' for i in reversed(range(32))])
    with pytest.raises(ValueError, match='Gene order differs'):
        workflow._check_training_parts(parts, config)
