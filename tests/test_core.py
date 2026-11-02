"""Deployment-state isolation and complete SELECT/STOP trajectory checks."""
import copy

import numpy as np
import pytest
import torch

from he_gene_rl.model import ContextPredictor
from he_gene_rl import core, ppo


def real_model_and_public():
    torch.manual_seed(77)
    model = ContextPredictor(feature_dim=3, gene_count=7, hidden_dim=8, use_neighbors=True).eval()
    # Some genes should be useful, others rejected; the real network remains
    # history-sensitive, so this checks the whole autoregressive trajectory.
    with torch.no_grad():
        model.state_head.bias.copy_(torch.tensor([
            [-1., -.5, 1.], [-1., 0., .8], [1., -.5, -1.],
            [.5, 0., .4], [.8, 0., -.3], [1., 0., -.4], [-.5, 0., .5],
        ]).reshape(-1))
    public = {'x': torch.randn(5, 3), 'neighbors': torch.randn(5, 2, 3),
              'relative_xy': torch.randn(5, 2, 2),
              'neighbor_mask': torch.ones(5, 2, dtype=torch.bool)}
    actor = ppo.ActorCritic(model.hidden_dim + 6 * model.gene_count + 1,
                            model.gene_count, hidden_dim=8, temperature=.25).eval()
    return model, public, actor


def test_real_context_initial_actor_matches_reference_every_action():
    model, public, actor = real_model_and_public()
    reference = core.predicted_sequence(model, public)
    initialized = core.predicted_sequence(model, public, actor, deterministic=True, record=True)
    assert bool(reference['selected_mask'].any())
    assert bool((~reference['selected_mask']).any())
    for key in ('selected_mask', 'sequence', 'predicted_values', 'stop_steps'):
        assert torch.equal(reference[key], initialized[key]), key
    for row in range(len(public['x'])):
        trace_rows = initialized['episode_index'] == row
        actions = initialized['actions'][trace_rows]
        steps = int(reference['stop_steps'][row])
        assert torch.equal(actions[:-1], reference['sequence'][row, :steps])
        assert actions[-1] == model.gene_count
        assert initialized['terminals'][trace_rows].tolist() == [False] * steps + [True]
    with pytest.raises(ValueError, match='RNA'):
        core.predicted_sequence(model, {**public, 'counts': torch.zeros(5, 7)}, actor)
    with pytest.raises(ValueError, match='reward'):
        core.predicted_sequence(model, {**public, 'reward': torch.zeros(5)}, actor)


class PoisonedRNAImages:
    """Real image assembler fields, while any RNA access fails immediately."""
    def __init__(self):
        rng = np.random.default_rng(22)
        self.features = rng.normal(size=(4, 3)).astype(np.float32)
        self.coordinates = rng.normal(size=(4, 2)).astype(np.float64)
        self.bank_features = np.concatenate((self.features, rng.normal(size=(3, 3)).astype(np.float32)))
        self.bank_coordinates = np.concatenate((self.coordinates, rng.normal(size=(3, 2))))
        self.neighbor_indices = np.array([[1, 4], [2, 5], [3, 6], [0, 4]])

    @property
    def counts(self):
        raise AssertionError('A deployment trajectory attempted to inspect RNA')


def test_collect_episode_image_only_and_posthoc_rewards_cannot_change_state():
    cohort = PoisonedRNAImages()
    model, _, actor = real_model_and_public()
    normalization = {'feature_mean': np.zeros(3, np.float32),
                     'feature_scale': np.ones(3, np.float32), 'coordinate_scale': 1.}
    ids = np.arange(4)
    trace = core.collect_episode(model, actor, cohort, ids, normalization, 'cpu',
                                 torch.Generator().manual_seed(919))
    assert bool((trace['actions'] < model.gene_count).any())
    frozen_trace = {key: value.clone() for key, value in trace.items()}
    zeros = np.zeros((4, model.gene_count), dtype=np.int64)
    positives = np.full_like(zeros, 4)
    empty_reward, _ = ppo.attach_rewards(trace, zeros)
    positive_reward, _ = ppo.attach_rewards(trace, positives)
    assert not torch.equal(empty_reward['rewards'], positive_reward['rewards'])
    for key in trace:
        assert torch.equal(trace[key], frozen_trace[key]), key
    for key in ('features', 'base_utility', 'selected', 'actions', 'old_log_prob', 'old_value'):
        assert torch.equal(empty_reward[key], positive_reward[key]), key
    repeated = core.collect_episode(model, actor, cohort, ids, normalization, 'cpu',
                                     torch.Generator().manual_seed(919))
    for key in trace:
        assert torch.equal(trace[key], repeated[key]), key


class CompletePanelToy(torch.nn.Module):
    def __init__(self, genes=4):
        super().__init__()
        self.gene_count, self.hidden_dim = genes, 1
        self.register_buffer('log_mean', torch.zeros(genes))
        self.register_buffer('log_scale', torch.ones(genes))
        self.calls = 0

    def encode(self, public):
        return {'center': public['x'][:, :1]}

    def score_encoded(self, encoded, values, selected):
        step = self.calls
        self.calls += 1
        n = len(values)
        utility = torch.arange(self.gene_count, 0, -1).to(values).repeat(n, 1)
        # First row STOPs immediately. Its later positive predictions cannot
        # cause it to reappear in the trace while the second row selects all.
        if step == 0:
            utility[0] = -1
        points = torch.arange(self.gene_count).to(values).repeat(n, 1) + step * 10 + .5
        return {'utility': utility,
                'probabilities': torch.full((n, self.gene_count, 3), 1/3, dtype=values.dtype),
                'log_count': points.log1p()}


def test_stop_absorbs_and_exhausted_panel_records_explicit_terminal_stop():
    model = CompletePanelToy()
    public = {'x': torch.zeros(2, 1), 'neighbors': torch.zeros(2, 1, 1),
              'relative_xy': torch.zeros(2, 1, 2),
              'neighbor_mask': torch.ones(2, 1, dtype=torch.bool)}
    actor = ppo.ActorCritic(1 + 6 * 4 + 1, 4, hidden_dim=8)
    trace = core.predicted_sequence(model, public, actor, deterministic=True, record=True)
    assert trace['stop_steps'].tolist() == [0, 4]
    assert trace['selected_mask'][0].sum() == 0
    assert trace['selected_mask'][1].all()
    assert trace['actions'][trace['episode_index'] == 0].tolist() == [4]
    assert trace['actions'][trace['episode_index'] == 1].tolist() == [0, 1, 2, 3, 4]
    assert trace['step_index'][trace['episode_index'] == 1].tolist() == [0, 1, 2, 3, 4]
    assert trace['terminals'][trace['episode_index'] == 1].tolist() == [False] * 4 + [True]
    torch.testing.assert_close(trace['predicted_values'][1], torch.tensor([.5, 11.5, 22.5, 33.5]))
    batch, metrics = ppo.attach_rewards(trace, np.array([[2, 2, 2, 2], [0, 1, 2, 3]]))
    assert metrics['cost'] == 4.5  # First cell FN4; second cell one committed FP.
    assert batch['rewards'][trace['episode_index'] == 1].tolist() == [-1., 0., 2., 2., 0.]
