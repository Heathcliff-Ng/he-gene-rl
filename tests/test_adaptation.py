"""Predicted-prefix isolation and exact source-adaptation recovery."""
from pathlib import Path

import numpy as np
import pytest
import torch

from he_gene_rl.model import ContextPredictor
from he_gene_rl import adaptation


def _public(n=3):
    return {'x': torch.arange(n, dtype=torch.float32)[:, None],
            'neighbors': torch.zeros(n, 2, 1), 'relative_xy': torch.zeros(n, 2, 2),
            'neighbor_mask': torch.ones(n, 2, dtype=torch.bool)}


class PrefixToy(torch.nn.Module):
    gene_count = 4

    def __init__(self):
        super().__init__()
        self.calls = []

    def encode(self, public):
        return public

    def score_encoded(self, encoded, values, selected):
        assert not torch.is_grad_enabled()
        assert not self.training
        call = len(self.calls)
        self.calls.append((values.clone(), selected.clone()))
        n = len(values)
        utility = torch.tensor([4., 3., 2., 1.]).repeat(n, 1)
        # First cell stops at the second call. A subsequent positive score
        # cannot reactivate it while other cells continue to their horizon.
        if call == 1:
            utility[0] = -1.
        point = torch.arange(4)[None].repeat(n, 1).float() + call * 10 + .25
        return {'utility': utility, 'log_count': point.log1p()}


def test_prefix_has_no_truth_input_uses_pre_action_continuous_prediction_and_stops():
    model = PrefixToy().train()
    public = _public()
    result = adaptation.predicted_prefix(model, public, torch.tensor([4, 4, 0]))
    assert model.training  # Preserve caller mode despite generation in eval.
    assert result['steps'].tolist() == [1, 4, 0]
    assert result['stopped'].tolist() == [True, False, False]
    assert result['selected_mask'][0].tolist() == [True, False, False, False]
    assert result['selected_mask'][1].all()
    torch.testing.assert_close(result['values'][0], torch.tensor([.25, 0., 0., 0.]))
    torch.testing.assert_close(result['values'][1], torch.tensor([.25, 11.25, 22.25, 33.25]))
    assert not result['values'].requires_grad
    with pytest.raises(ValueError, match='public'):
        adaptation.predicted_prefix(PrefixToy(), {**public, 'counts': torch.zeros(3, 4)}, [1, 1, 1])


class SyntheticCohort:
    def __init__(self, prefix, n, seed):
        rng = np.random.default_rng(seed)
        self.features = rng.normal(size=(n, 3)).astype(np.float32)
        self.neighbors = rng.normal(size=(n, 2, 3)).astype(np.float32)
        self.xy = rng.normal(size=(n, 2, 2)).astype(np.float32)
        self.counts = rng.integers(0, 5, size=(n, 5)).astype(np.float32)
        self.cell_ids = np.array([f'{prefix}_{i}' for i in range(n)])

    def __len__(self):
        return len(self.features)


def synthetic_public(cohort, indices, normalization, device):
    return {'x': torch.as_tensor(cohort.features[indices], device=device),
            'neighbors': torch.as_tensor(cohort.neighbors[indices], device=device),
            'relative_xy': torch.as_tensor(cohort.xy[indices], device=device),
            'neighbor_mask': torch.ones(len(indices), 2, dtype=torch.bool, device=device)}


def synthetic_monitor(model, cohort, normalization, settings, device):
    public = synthetic_public(cohort, np.arange(len(cohort)), {}, device)
    result = adaptation.predicted_prefix(model, public, torch.full((len(cohort),), model.gene_count))
    selected = result['selected_mask'].numpy()
    fp = (selected & (cohort.counts == 0)).sum(1)
    fn = (~selected & (cohort.counts >= 2)).sum(1)
    return {'cost': float((fp + 2 * fn).mean()), 'fp': float(fp.mean()), 'fn': float(fn.mean())}


SETTINGS = {'learning_rate': .003, 'weight_decay': .0001, 'min_epochs': 2,
            'max_epochs': 2, 'monitor_every': 1, 'patience': 2, 'batch_size': 4,
            'history_budgets': [0, 1, 2, 4], 'count_weight': .1, 'gradient_clip': 5.}


def _teacher(tmp_path, train):
    torch.manual_seed(42)
    model = ContextPredictor(feature_dim=3, gene_count=5, hidden_dim=8, use_neighbors=True)
    model.fit_count_statistics(train.counts)
    path = tmp_path / 'teacher.pt'
    torch.save({'model_config': model.get_config(), 'state_dict': model.state_dict()}, path)
    return path


def test_hidden_counts_cannot_change_generated_prefix(tmp_path):
    cohort = SyntheticCohort('train', 4, 10)
    teacher = _teacher(tmp_path, cohort)
    model, _ = adaptation.load_model(teacher)
    public = synthetic_public(cohort, np.arange(4), {}, 'cpu')
    before = adaptation.predicted_prefix(model, public, [4, 2, 1, 0])
    cohort.counts[:] = 999.
    after = adaptation.predicted_prefix(model, synthetic_public(cohort, np.arange(4), {}, 'cpu'), [4, 2, 1, 0])
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_resume_exact_and_supervised_targets_match(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    train = SyntheticCohort('train', 8, 10)
    monitor = SyntheticCohort('monitor', 4, 11)
    teacher = _teacher(tmp_path, train)
    teacher_sha = adaptation.sha256(teacher)
    monkeypatch.setattr(adaptation, 'public_from_cohort', synthetic_public)
    monkeypatch.setattr(adaptation, 'monitor_adapter', synthetic_monitor)
    actual_loss = adaptation.prediction_loss
    calls = []

    def checked_loss(output, truth, target, **kwargs):
        calls.append((truth.detach().clone(), target.detach().clone()))
        if len(calls) % 2 == 0:
            assert torch.equal(calls[-1][0], calls[-2][0])
            assert torch.equal(calls[-1][1], calls[-2][1])
        return actual_loss(output, truth, target, **kwargs)

    monkeypatch.setattr(adaptation, 'prediction_loss', checked_loss)
    full_dir, partial_dir = tmp_path / 'full', tmp_path / 'partial'
    full_model, full_summary = adaptation.fit_adapter(teacher, train, monitor, {}, SETTINGS,
                                                       full_dir, 42, 'cpu')
    full_rng = torch.rand(5)
    actual_save = adaptation.save_checkpoint

    class Interrupted(RuntimeError):
        pass

    def interrupt_after_checkpoint(path, payload):
        actual_save(path, payload)
        if Path(path).name == 'last.pt' and payload['epoch'] == 1:
            raise Interrupted('Checkpoint is durable; progress write has not happened')

    monkeypatch.setattr(adaptation, 'save_checkpoint', interrupt_after_checkpoint)
    with pytest.raises(Interrupted):
        adaptation.fit_adapter(teacher, train, monitor, {}, SETTINGS, partial_dir, 42, 'cpu')
    monkeypatch.setattr(adaptation, 'save_checkpoint', actual_save)
    torch.manual_seed(8374)
    _ = torch.rand(41)
    recovered, recovered_summary = adaptation.fit_adapter(teacher, train, monitor, {}, SETTINGS,
                                                           partial_dir, 42, 'cpu')
    assert torch.equal(full_rng, torch.rand(5))
    for key, value in full_model.state_dict().items():
        assert torch.equal(value, recovered.state_dict()[key]), key
    full_last = torch.load(full_dir / 'last.pt', map_location='cpu', weights_only=False)
    recovered_last = torch.load(partial_dir / 'last.pt', map_location='cpu', weights_only=False)
    for key, value in full_last['state_dict'].items():
        assert torch.equal(value, recovered_last['state_dict'][key]), key
    assert torch.equal(full_last['generator_state'], recovered_last['generator_state'])
    for key in ('best_epoch', 'last_epoch', 'updates', 'best_monitor'):
        assert full_summary[key] == recovered_summary[key]
    assert adaptation.sha256(teacher) == teacher_sha
    # Initial weights can remain selected if adaptation fails to improve.
    assert full_summary['best_epoch'] >= 0
    for original, recovered_record in zip(full_summary['records'], recovered_summary['records']):
        for key in ('epoch', 'loss', 'mean_generated_history_steps', 'monitor'):
            if key in original:
                assert original[key] == recovered_record[key]
