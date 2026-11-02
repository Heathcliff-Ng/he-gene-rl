"""Input isolation and matched-arm checks for contextual conditional prediction."""
import pytest
import torch

from he_gene_rl.model import ContextPredictor, expected_utility, prediction_loss, state_labels


def inputs(n=3, genes=5, feature_dim=6, neighbors=4):
    torch.manual_seed(97)
    public = {'x': torch.randn(n, feature_dim),
              'neighbors': torch.randn(n, neighbors, feature_dim),
              'relative_xy': torch.randn(n, neighbors, 2),
              'neighbor_mask': torch.ones(n, neighbors, dtype=torch.bool)}
    counts = torch.tensor([[0., 1., 2., 4., 0.]]).expand(n, genes).clone()
    mask = torch.zeros(n, genes, dtype=torch.bool)
    mask[:, :2] = True
    return public, counts, mask


def model(neighbors=True):
    torch.manual_seed(81)
    return ContextPredictor(6, 5, 12, use_neighbors=neighbors).eval()


def test_hidden_values_and_extra_labels_cannot_affect_predictions():
    public, counts, mask = inputs()
    predictor = model()
    a = predictor(public, counts, mask)
    changed = counts.clone()
    changed[~mask] = torch.nan
    b = predictor(public, changed, mask)
    torch.testing.assert_close(a['state_logits'], b['state_logits'], rtol=0, atol=0)
    torch.testing.assert_close(a['log_count'], b['log_count'], rtol=0, atol=0)
    with pytest.raises(ValueError, match='Public inputs'):
        predictor({**public, 'counts': counts}, counts, mask)


def test_identity_only_is_not_an_observed_zero_and_ignores_withheld_values():
    public, counts, mask = inputs()
    predictor = model()
    missing = torch.zeros_like(mask)
    identity_only = predictor(public, torch.full_like(counts, torch.nan), mask, missing)
    different_withheld = predictor(public, torch.full_like(counts, 999.), mask, missing)
    torch.testing.assert_close(identity_only['state_logits'], different_withheld['state_logits'], rtol=0, atol=0)
    observed_zeros = predictor(public, torch.zeros_like(counts), mask)
    no_history = predictor(public, torch.zeros_like(counts), missing)
    assert not torch.equal(identity_only['state_logits'], observed_zeros['state_logits'])
    assert not torch.equal(identity_only['state_logits'], no_history['state_logits'])
    with pytest.raises(ValueError, match='unselected'):
        predictor(public, counts, missing, mask)


def test_arms_have_identical_initialization_and_center_ignores_every_neighbor():
    public, counts, mask = inputs()
    center, spatial = model(False), model(True)
    assert center.get_config() == {**spatial.get_config(), 'use_neighbors': False}
    for key, tensor in center.state_dict().items():
        torch.testing.assert_close(tensor, spatial.state_dict()[key], rtol=0, atol=0)
    assert sum(p.numel() for p in center.parameters()) == sum(p.numel() for p in spatial.parameters())
    base = center(public, counts, mask)
    corrupted = {**public, 'neighbors': torch.full_like(public['neighbors'], torch.nan),
                 'relative_xy': torch.full_like(public['relative_xy'], torch.nan)}
    same = center(corrupted, counts, mask)
    torch.testing.assert_close(base['state_logits'], same['state_logits'], rtol=0, atol=0)
    assert (base['neighbor_attention'] == 0).all()
    assert (base['self_attention'] == 1).all()


def test_neighbor_evidence_positions_and_history_can_change_attention():
    public, counts, mask = inputs()
    predictor = model()
    before = predictor(public, counts, mask)
    changed = predictor({**public, 'neighbors': -public['neighbors'],
                         'relative_xy': public['relative_xy'] + 3.}, counts, mask)
    assert not torch.equal(before['state_logits'], changed['state_logits'])
    assert not torch.equal(before['neighbor_attention'], changed['neighbor_attention'])
    empty = predictor(public, counts, torch.zeros_like(mask))
    assert not torch.equal(before['neighbor_attention'], empty['neighbor_attention'])
    torch.testing.assert_close(before['neighbor_attention'].sum(1) + before['self_attention'], torch.ones(len(counts)))


def test_masked_neighbor_nan_and_neighbor_permutation_are_inert():
    public, counts, mask = inputs()
    predictor = model()
    public['neighbor_mask'][:, -1] = False
    clean = predictor(public, counts, mask)
    public['neighbors'][:, -1] = torch.nan
    public['relative_xy'][:, -1] = torch.nan
    changed = predictor(public, counts, mask)
    torch.testing.assert_close(clean['state_logits'], changed['state_logits'], rtol=0, atol=0)
    order = torch.tensor([2, 0, 3, 1])
    permuted = {key: value[:, order] if key != 'x' else value for key, value in public.items()}
    result = predictor(permuted, counts, mask)
    torch.testing.assert_close(changed['state_logits'], result['state_logits'], rtol=1e-5, atol=1e-7)


def test_three_outcome_utility_and_source_statistics():
    probabilities = torch.eye(3).unsqueeze(0)
    torch.testing.assert_close(expected_utility(probabilities), torch.tensor([[-1., 0., 2.]]))
    torch.testing.assert_close(state_labels(torch.tensor([[0., 1., 2., 20.]])), torch.tensor([[0, 1, 2, 2]]))
    _, counts, _ = inputs()
    predictor = model()
    predictor.fit_count_statistics(counts)
    torch.testing.assert_close(predictor.log_mean, counts.log1p().mean(0))
    assert torch.isfinite(predictor.count_head.bias).all()
    with pytest.raises(ValueError, match='integers'):
        predictor.fit_count_statistics(counts + .1)


def test_loss_uses_only_common_target_scope_and_raw_one_is_neutral_for_cost():
    logits = torch.tensor([[[0., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]], requires_grad=True)
    prediction = {'state_logits': logits, 'log_count': torch.zeros(1, 4, requires_grad=True)}
    truth = torch.tensor([[0., 1., 2., 4.]])
    targets = torch.tensor([[False, True, True, True]])
    loss, metrics = prediction_loss(prediction, truth, targets)
    assert metrics['n_targets'].item() == 3
    assert metrics['n_detection_targets'].item() == 2
    # raw1 chosen or not chosen changes neither FP nor FN. Gene3 is missed.
    assert metrics['cost'].item() == 2
    loss.backward()
    assert (logits.grad[:, 0] == 0).all()
    assert (prediction['log_count'].grad[:, :2] == 0).all()
    changed = truth.clone()
    changed[:, 0] = 777
    again, _ = prediction_loss(prediction, changed, targets)
    torch.testing.assert_close(loss, again)
    empty, fields = prediction_loss(prediction, truth, torch.zeros_like(targets))
    assert empty.item() == 0 and fields['cost'].item() == 0


def test_checkpoint_config_reconstructs_model_and_cached_encoding_matches_forward():
    public, counts, mask = inputs()
    original = model(True)
    original.fit_count_statistics(counts)
    reconstructed = ContextPredictor(**original.get_config())
    reconstructed.load_state_dict(original.state_dict())
    direct = original(public, counts, mask)
    cached = reconstructed.score_encoded(reconstructed.encode(public), counts, mask)
    for key in ('state_logits', 'log_count', 'utility'):
        torch.testing.assert_close(direct[key], cached[key], rtol=0, atol=0)
