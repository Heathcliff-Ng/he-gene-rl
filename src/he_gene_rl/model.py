"""Matched center/neighbor predictors with explicit observed-count semantics.

Both arms instantiate identical modules. The center arm masks every neighbor
token; the self token keeps the context path usable without spatial evidence.
Selected identities and revealed values have separate masks: withholding a
measurement never turns it into an observed biological zero.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


PUBLIC_KEYS = ('x', 'neighbors', 'relative_xy', 'neighbor_mask')


def public_batch(batch):
    """Explicitly project a training/evaluation batch onto model-public inputs."""
    return {key: batch[key] for key in PUBLIC_KEYS}


def state_labels(counts):
    """raw0 -> 0, raw1 -> 1, raw>=2 -> 2; raw1 is a neutral outcome."""
    return torch.where(counts == 0, 0, torch.where(counts == 1, 1, 2)).long()


def expected_utility(probabilities):
    """Exact immediate reward expectation for reward(-1, 0, +2)."""
    if probabilities.ndim < 2 or probabilities.shape[-1] != 3:
        raise ValueError('Three outcome probabilities are required')
    return 2. * probabilities[..., 2] - probabilities[..., 0]


class ContextPredictor(nn.Module):
    """Conditional classifier with center/history-conditioned spatial attention.

The output is a marginal three-outcome distribution for each gene, not a
joint generative count model. Neighbor features contain image information
only. Neighborhood variance summarizes heterogeneous tokens without assuming
that every adjacent cell shares the center's expression state.
"""
    def __init__(self, feature_dim=1024, gene_count=313, hidden_dim=128,
                 use_neighbors=False):
        super().__init__()
        if min(feature_dim, gene_count, hidden_dim) < 1:
            raise ValueError('Model dimensions must be positive')
        self.feature_dim = int(feature_dim)
        self.gene_count = int(gene_count)
        self.hidden_dim = int(hidden_dim)
        self.use_neighbors = bool(use_neighbors)
        self.register_buffer('log_mean', torch.zeros(self.gene_count))
        self.register_buffer('log_scale', torch.ones(self.gene_count))
        self.image = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_dim),
                                   nn.LayerNorm(self.hidden_dim), nn.SiLU())
        self.position = nn.Sequential(nn.Linear(2, self.hidden_dim), nn.SiLU(),
                                      nn.Linear(self.hidden_dim, self.hidden_dim))
        self.contrast = nn.Linear(self.feature_dim, self.hidden_dim, bias=False)
        self.history = nn.Sequential(nn.Linear(3 * self.gene_count + 2, self.hidden_dim),
                                     nn.LayerNorm(self.hidden_dim), nn.SiLU())
        self.query = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                                    nn.LayerNorm(self.hidden_dim), nn.SiLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU())
        self.state_head = nn.Linear(self.hidden_dim, 3 * self.gene_count)
        self.count_head = nn.Linear(self.hidden_dim, self.gene_count)
        # Near-prior starts retain input sensitivity and train both arms from
        # the same weights. Source-only bias fitting is an explicit later call.
        nn.init.normal_(self.state_head.weight, std=.01)
        nn.init.zeros_(self.state_head.bias)
        nn.init.normal_(self.count_head.weight, std=.01)
        nn.init.zeros_(self.count_head.bias)

    def get_config(self):
        return {'feature_dim': self.feature_dim, 'gene_count': self.gene_count,
                'hidden_dim': self.hidden_dim, 'use_neighbors': self.use_neighbors}

    @torch.no_grad()
    def fit_count_statistics(self, counts):
        """Fit buffers and initial head biases on source TRAIN counts only.

        The caller owns split enforcement; this method never reads a cohort
        loader or target-section data. Call before optimization, not on resume.
        """
        counts = torch.as_tensor(counts, device=self.log_mean.device,
                                 dtype=self.log_mean.dtype)
        if counts.ndim != 2 or counts.shape[1] != self.gene_count or not len(counts):
            raise ValueError('Source counts must be nonempty [cells, genes]')
        _validate_counts(counts)
        values = counts.log1p()
        self.log_mean.copy_(values.mean(0))
        self.log_scale.copy_(values.std(0, unbiased=False).clamp_min(.25))
        frequencies = torch.stack([(counts == 0).sum(0), (counts == 1).sum(0),
                                   (counts >= 2).sum(0)], dim=-1).to(counts.dtype) + 1.
        probabilities = frequencies / frequencies.sum(-1, keepdim=True)
        self.state_head.bias.copy_(probabilities.log().reshape(-1))
        valid = counts != 1
        mean_log = (values * valid).sum(0) / valid.sum(0).clamp_min(1)
        # Inverse softplus of a nonzero floor gives a finite initial mean.
        mean_log = mean_log.clamp_min(1e-4)
        self.count_head.bias.copy_(mean_log + torch.log(-torch.expm1(-mean_log)))

    def _validate_public(self, public):
        if set(public) != set(PUBLIC_KEYS):
            raise ValueError('Public inputs must contain only x, neighbors, relative_xy, neighbor_mask')
        x, neighbors, xy, mask = (public[key] for key in PUBLIC_KEYS)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError('Center features must be [cells, feature_dim]')
        if neighbors.ndim != 3 or neighbors.shape[0] != len(x) or neighbors.shape[2] != self.feature_dim:
            raise ValueError('Neighbor features must be [cells, neighbors, feature_dim]')
        if xy.shape != (*neighbors.shape[:2], 2) or mask.shape != neighbors.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('Relative coordinates and boolean neighbor mask must align')
        if not bool(torch.isfinite(x).all()):
            raise ValueError('Center features must be finite')
        return x, neighbors, xy, mask

    def encode(self, public):
        """Cache image/position encodings while comparing observation controls."""
        x, neighbors, xy, supplied_mask = self._validate_public(public)
        mask = supplied_mask if self.use_neighbors else torch.zeros_like(supplied_mask)
        # Invalid/masked neighbor values may even be NaN. Mask them before
        # subtraction, projection, attention, or any other arithmetic.
        clean_neighbors = torch.where(mask[..., None], neighbors, torch.zeros_like(neighbors))
        clean_xy = torch.where(mask[..., None], xy, torch.zeros_like(xy))
        if not bool(torch.isfinite(clean_neighbors).all()) or not bool(torch.isfinite(clean_xy).all()):
            raise ValueError('Available neighbor features and coordinates must be finite')
        center = self.image(x)
        neighbor_features = self.image(clean_neighbors)
        contrast = torch.where(mask[..., None], clean_neighbors - x[:, None],
                               torch.zeros_like(clean_neighbors))
        neighbor_tokens = neighbor_features + self.position(clean_xy) + self.contrast(contrast)
        self_position = self.position(x.new_zeros((len(x), 2)))
        tokens = torch.cat(((center + self_position)[:, None], neighbor_tokens), dim=1)
        feature_tokens = torch.cat((center[:, None], neighbor_features), dim=1)
        token_mask = torch.cat((torch.ones((len(x), 1), dtype=torch.bool, device=x.device), mask), dim=1)
        return {'center': center, 'keys': self.key(tokens), 'values': self.value(tokens),
                'feature_tokens': feature_tokens, 'token_mask': token_mask}

    def score_encoded(self, encoded, values, observed_mask, value_mask=None):
        shape = (len(encoded['center']), self.gene_count)
        if values.shape != shape or observed_mask.shape != shape or observed_mask.dtype != torch.bool:
            raise ValueError('Observation values and boolean selected mask must be [cells, genes]')
        value_mask = observed_mask if value_mask is None else value_mask
        if value_mask.shape != shape or value_mask.dtype != torch.bool:
            raise ValueError('Value availability mask must be boolean [cells, genes]')
        if bool((value_mask & ~observed_mask).any()):
            raise ValueError('A value cannot be available for an unselected gene')
        clean = torch.where(value_mask, values, torch.zeros_like(values))
        if not bool(torch.isfinite(clean).all()) or bool((clean < 0).any()):
            raise ValueError('Available count values must be finite and nonnegative')
        normalized = (clean.log1p() - self.log_mean) / self.log_scale
        normalized = torch.where(value_mask, normalized, torch.zeros_like(normalized))
        identity = observed_mask.to(clean.dtype)
        available = value_mask.to(clean.dtype)
        history = self.history(torch.cat((normalized, identity, available,
                                          identity.mean(1, keepdim=True),
                                          available.mean(1, keepdim=True)), dim=1))
        center = encoded['center']
        query = self.query(torch.cat((center, history), dim=1))
        logits = (encoded['keys'] * query[:, None]).sum(-1) / math.sqrt(self.hidden_dim)
        attention = logits.masked_fill(~encoded['token_mask'], -torch.inf).softmax(1)
        message = (attention[..., None] * encoded['values']).sum(1)
        mean = (attention[..., None] * encoded['feature_tokens']).sum(1)
        variance = (attention[..., None] * (encoded['feature_tokens'] - mean[:, None]).square()).sum(1)
        spread = (variance + 1e-6).sqrt()
        hidden = self.fusion(torch.cat((center, history, message, spread), dim=1))
        state_logits = self.state_head(hidden).reshape(-1, self.gene_count, 3)
        probabilities = state_logits.softmax(-1)
        return {'state_logits': state_logits, 'probabilities': probabilities,
                'utility': expected_utility(probabilities),
                'log_count': F.softplus(self.count_head(hidden)),
                'neighbor_attention': attention[:, 1:], 'self_attention': attention[:, 0],
                'context_variance': variance}

    def forward(self, public, values, observed_mask, value_mask=None):
        return self.score_encoded(self.encode(public), values, observed_mask, value_mask)


def _validate_counts(counts):
    if (not bool(torch.isfinite(counts).all()) or bool((counts < 0).any())
            or bool((counts != counts.floor()).any())):
        raise ValueError('Counts must be finite nonnegative integers')


def prediction_loss(output, counts, target_mask, count_weight=.1):
    """Return scalar fitting loss and metrics per cell on a fixed target scope.

    State CE models raw1 as its own outcome so utility uses unconditional
    P(raw0)/P(raw>=2). Binary detection and log-count regression exclude raw1.
    Remaining-set cost is an unnormalized per-cell count, whereas the three
    fitting metrics are means over their respective valid target genes.
    """
    logits, log_count = output['state_logits'], output['log_count']
    if logits.ndim != 3 or logits.shape[-1] != 3 or counts.shape != logits.shape[:2] or log_count.shape != counts.shape:
        raise ValueError('Predictions and counts must align on [cells, genes]')
    if target_mask.shape != counts.shape or target_mask.dtype != torch.bool:
        raise ValueError('Target scope must be a boolean [cells, genes] mask')
    if not math.isfinite(float(count_weight)) or count_weight < 0:
        raise ValueError('Count loss weight must be finite and nonnegative')
    _validate_counts(counts)
    labels = state_labels(counts)
    ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
    n_targets = target_mask.sum(1)
    state_ce = (ce * target_mask).sum(1) / n_targets.clamp_min(1)
    detection_mask = target_mask & (counts != 1)
    n_detection = detection_mask.sum(1)
    binary_logits = logits[..., 2] - logits[..., 0]
    bce = F.binary_cross_entropy_with_logits(binary_logits, (counts >= 2).to(logits.dtype), reduction='none')
    detection_bce = (bce * detection_mask).sum(1) / n_detection.clamp_min(1)
    log_mse = ((log_count - counts.log1p()).square() * detection_mask).sum(1) / n_detection.clamp_min(1)
    per_cell = state_ce + float(count_weight) * log_mse
    valid_cells = n_targets > 0
    loss = (per_cell * valid_cells).sum() / valid_cells.sum().clamp_min(1)
    selected = expected_utility(logits.softmax(-1)) > 0
    fp = (selected & (counts == 0) & target_mask).sum(1)
    fn = (~selected & (counts >= 2) & target_mask).sum(1)
    return loss, {'state_ce': state_ce, 'detection_bce': detection_bce,
                  'log_count_mse': log_mse, 'cost': fp + 2 * fn,
                  'n_targets': n_targets, 'n_detection_targets': n_detection}
