"""Checkpoint recovery and matched-observation training invariants."""
from pathlib import Path

import numpy as np
import pytest
import torch

from he_gene_rl.model import ContextPredictor
from he_gene_rl import pretraining as training


class SyntheticCohort:
    def __init__(self, prefix, n, seed):
        rng = np.random.default_rng(seed)
        self.features = rng.normal(size=(n, 3)).astype(np.float32)
        self.neighbors = rng.normal(size=(n, 2, 3)).astype(np.float32)
        self.xy = rng.normal(size=(n, 2, 2)).astype(np.float32)
        self.counts = rng.integers(0, 4, size=(n, 40)).astype(np.float32)
        self.cell_ids = np.array([f'{prefix}_{i}' for i in range(n)])
        self.tiles = np.arange(n) // 2

    def __len__(self):
        return len(self.counts)

    def batch(self, indices, device, normalization):
        return {'x': torch.as_tensor(self.features[indices], device=device),
                'neighbors': torch.as_tensor(self.neighbors[indices], device=device),
                'relative_xy': torch.as_tensor(self.xy[indices], device=device),
                'neighbor_mask': torch.ones(len(indices), 2, device=device, dtype=torch.bool),
                'counts': torch.as_tensor(self.counts[indices], device=device)}


SETTINGS = {'learning_rate': .003, 'weight_decay': .0001, 'min_epochs': 2,
            'max_epochs': 2, 'monitor_every': 1, 'patience': 2, 'batch_size': 4,
            'history_budgets': [0, 4, 8, 16, 32], 'identity_only_fraction': .15,
            'count_weight': .1, 'gradient_clip': 5.}


def _model(train):
    torch.manual_seed(42)
    model = ContextPredictor(feature_dim=3, gene_count=40, hidden_dim=8, use_neighbors=True)
    model.fit_count_statistics(train.counts)
    return model


def test_epoch_checkpoint_resume_exactly_matches_uninterrupted_fit(tmp_path, monkeypatch):
    train = SyntheticCohort('train', 8, 10)
    monitor = SyntheticCohort('monitor', 4, 11)
    assert set(train.cell_ids).isdisjoint(monitor.cell_ids)
    # Ensure both loss calls score exactly the same remaining genes. This
    # prevents an apparent conditional improvement caused by target removal.
    original_loss = training.prediction_loss
    calls = []

    def checked_loss(output, counts, target, **kwargs):
        calls.append((counts.detach().clone(), target.detach().clone()))
        if len(calls) % 2 == 0:
            assert torch.equal(calls[-1][0], calls[-2][0])
            assert torch.equal(calls[-1][1], calls[-2][1])
        return original_loss(output, counts, target, **kwargs)

    monkeypatch.setattr(training, 'prediction_loss', checked_loss)
    full_dir, interrupted_dir = tmp_path / 'full', tmp_path / 'interrupted'
    full_model, full_summary = training.fit(_model(train), train, monitor, {}, SETTINGS,
                                           full_dir, seed=42, device='cpu')
    full_rng_after = torch.rand(5)
    original_save = training.save_checkpoint

    class SimulatedInterruption(RuntimeError):
        pass

    def interrupt_after_epoch_one(path, payload):
        original_save(path, payload)
        if Path(path).name == 'last.pt' and payload['epoch'] == 1:
            raise SimulatedInterruption('checkpoint written, progress file not yet written')

    monkeypatch.setattr(training, 'save_checkpoint', interrupt_after_epoch_one)
    with pytest.raises(SimulatedInterruption):
        training.fit(_model(train), train, monitor, {}, SETTINGS,
                     interrupted_dir, seed=42, device='cpu')
    assert not (interrupted_dir / 'summary.json').exists()
    monkeypatch.setattr(training, 'save_checkpoint', original_save)
    resumed = _model(train)
    torch.manual_seed(999999)
    _ = torch.rand(53)  # Unrelated process work cannot change resumed updates.
    resumed_model, resumed_summary = training.fit(resumed, train, monitor, {}, SETTINGS,
                                                 interrupted_dir, seed=42, device='cpu')
    assert torch.equal(torch.rand(5), full_rng_after)
    full_last = torch.load(full_dir / 'last.pt', map_location='cpu', weights_only=False)
    resumed_last = torch.load(interrupted_dir / 'last.pt', map_location='cpu', weights_only=False)
    for key, value in full_last['state_dict'].items():
        assert torch.equal(value, resumed_last['state_dict'][key]), key
    assert torch.equal(full_last['generator_state'], resumed_last['generator_state'])
    for key, value in full_model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[key]), key
    for key in ('best_epoch', 'last_epoch', 'updates', 'best_monitor'):
        assert resumed_summary[key] == full_summary[key]
    for first, second in zip(full_summary['records'], resumed_summary['records']):
        assert first['epoch'] == second['epoch']
        if 'loss' in first:
            assert first['loss'] == second['loss']
        if 'monitor' in first:
            assert first['monitor'] == second['monitor']
    assert len(calls) == 16  # Four minibatches per completed 2-epoch path.


def test_history_identity_sampling_never_uses_count_outcomes():
    counts = torch.arange(256 * 40, dtype=torch.float32).reshape(256, 40) % 4
    changed = torch.flip(counts, (1,)) + 1
    kwargs = {'budgets': [0, 4, 8, 16, 32], 'identity_only_fraction': .5}
    a = training.sample_observations(counts, generator=torch.Generator().manual_seed(8), **kwargs)
    b = training.sample_observations(changed, generator=torch.Generator().manual_seed(8), **kwargs)
    values, observed, available = a
    assert torch.equal(observed, b[1]) and torch.equal(available, b[2])
    assert not (available & ~observed).any()
    assert (values[~available] == 0).all()
    assert torch.equal(values[available], counts[available])
    assert not torch.equal(values[available], b[0][available])
    assert any((observed.sum(1) > 0) & (available.sum(1) == 0))
