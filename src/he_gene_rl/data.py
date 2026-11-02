"""Portable image banks and expression targets with explicit identity joins.

Paths are resolved relative to a user-supplied manifest, never this package.
Image neighbors may include held-out image nodes; their RNA is never loaded.
Use ``require_counts=False`` for inference without reading expression arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch


_HASH_CACHE = {}
_BANK_CACHE = {}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path, expected, inputs):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    digest = _HASH_CACHE.get(key)
    if digest is None:
        digest = _HASH_CACHE[key] = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f'Input checksum mismatch: {path}')
    inputs[str(path)] = digest
    return path


def _identities(values, label):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in 'US':
        raise ValueError(f'{label} must be a nonempty string vector (no object arrays)')
    values = values.astype(str)
    if np.any(np.char.str_len(values) == 0) or len(np.unique(values)) != len(values):
        raise ValueError(f'{label} must contain unique nonempty identities')
    return values


def neighbor_indices(coordinates, query_rows, neighbors=8, tree=None):
    """Find nearest image-position neighbors, excluding the query cell itself."""
    xy = np.asarray(coordinates, dtype=np.float64)
    rows = np.asarray(query_rows)
    if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all()
            or rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
            or np.any(rows < 0) or np.any(rows >= len(xy))
            or isinstance(neighbors, bool) or int(neighbors) != neighbors
            or not 1 <= int(neighbors) < len(xy)):
        raise ValueError('Finite [cells,2] positions, valid query rows and neighbor count required')
    k = int(neighbors)
    tree = cKDTree(xy) if tree is None else tree
    distances, candidates = tree.query(xy[rows], k=k + 1, workers=1)
    result = np.empty((len(rows), k), dtype=np.int64)
    for i, row in enumerate(rows):
        order = np.lexsort((candidates[i], distances[i]))
        choices = candidates[i, order]
        result[i] = choices[choices != row][:k]
    if np.any(result == rows[:, None]):
        raise AssertionError('Self entered an explicit neighborhood')
    return result


@dataclass
class ContextCohort:
    section: str
    part: str
    features: np.ndarray
    counts: np.ndarray | None
    cell_ids: np.ndarray
    row_indices: np.ndarray
    coordinates: np.ndarray
    tiles: np.ndarray
    gene_ids: np.ndarray
    neighbor_indices: np.ndarray
    bank_features: np.ndarray
    bank_coordinates: np.ndarray
    metadata: dict

    def __len__(self):
        return len(self.features)

    @property
    def feature_dim(self):
        return self.features.shape[1]

    @property
    def gene_count(self):
        return len(self.gene_ids)

    def batch(self, indices, device='cpu', normalization=None):
        """Supervised batch; inference must use ``public_from_features`` instead."""
        if self.counts is None:
            raise ValueError('Expression targets are unavailable in this image-only cohort')
        indices = _batch_indices(self, indices)
        public = public_from_features(self, indices, normalization, device)
        public['counts'] = torch.as_tensor(
            np.asarray(self.counts[indices], dtype=np.float32), device=device)
        return public


def _batch_indices(cohort, indices):
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().numpy()
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
            or np.any(indices < 0) or np.any(indices >= len(cohort.features))):
        raise ValueError('Batch indices must be a nonempty in-range integer vector')
    return indices


def _normalization(cohort, normalization):
    if normalization is None:
        raise ValueError('Pass normalization fitted on source TRAIN, including at test time')
    mean = np.asarray(normalization['feature_mean'], dtype=np.float32)
    scale = np.asarray(normalization['feature_scale'], dtype=np.float32)
    coordinate_scale = float(normalization['coordinate_scale'])
    if (mean.shape != (cohort.features.shape[1],) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0) or not np.isfinite(coordinate_scale) or coordinate_scale <= 0):
        raise ValueError('Invalid source normalization')
    return mean, scale, coordinate_scale


def public_from_features(cohort, indices, normalization, device='cpu'):
    """Construct policy-public inputs without accessing ``cohort.counts``."""
    indices = _batch_indices(cohort, indices)
    mean, scale, coordinate_scale = _normalization(cohort, normalization)
    center = np.asarray(cohort.features[indices], dtype=np.float32)
    neighbor_rows = cohort.neighbor_indices[indices]
    valid = neighbor_rows >= 0
    safe_rows = np.maximum(neighbor_rows, 0)
    neighbor = np.asarray(cohort.bank_features[safe_rows], dtype=np.float32)
    xy = ((cohort.bank_coordinates[safe_rows] - cohort.coordinates[indices, None])
          / coordinate_scale).astype(np.float32)
    return {'x': torch.as_tensor((center - mean) / scale, device=device),
            'neighbors': torch.as_tensor((neighbor - mean) / scale, device=device),
            'relative_xy': torch.as_tensor(xy, device=device),
            'neighbor_mask': torch.as_tensor(valid, device=device)}


def fit_normalization(cohort, chunk_size=4096):
    """Fit image normalization using source TRAIN centers only."""
    if cohort.part != 'train':
        raise ValueError('Normalization may only be fitted on source TRAIN cells')
    if chunk_size < 1 or not len(cohort):
        raise ValueError('Positive chunk size and nonempty source train required')
    total = np.zeros(cohort.feature_dim, dtype=np.float64)
    squared = total.copy()
    for start in range(0, len(cohort), chunk_size):
        block = np.asarray(cohort.features[start:start + chunk_size], dtype=np.float64)
        total += block.sum(0)
        squared += np.einsum('ij,ij->j', block, block)
    mean = total / len(cohort)
    variance = np.maximum(squared / len(cohort) - mean ** 2, 0.)
    delta = cohort.bank_coordinates[cohort.neighbor_indices[:, 0]] - cohort.coordinates
    distance = np.linalg.norm(delta, axis=1)
    positive = distance[distance > 0]
    coordinate_scale = max(float(np.median(positive)) if len(positive) else 1., 1e-6)
    return {'feature_mean': mean.astype(np.float32),
            'feature_scale': np.sqrt(variance).clip(1e-6).astype(np.float32),
            'coordinate_scale': coordinate_scale}


def load_cohort(manifest, section, part, neighbors=8, max_cells=None, require_counts=True):
    """Load one split from a portable manifest and verify its bank identities.

    ``require_counts=False`` does not access any expression array, even when
    one is present. ``tiles`` may be omitted; such cohorts cannot support a
    spatial uncertainty interval. File hashes are recorded for reproducibility.
    """
    if max_cells is not None and (isinstance(max_cells, bool)
            or int(max_cells) != max_cells or max_cells < 1):
        raise ValueError('max_cells must be a positive integer or None')
    inputs = {}
    manifest_path = _checked(manifest, None, inputs)
    spec = json.loads(manifest_path.read_text(encoding='utf-8'))
    if spec.get('format') != 'he-gene-rl-data-v1':
        raise ValueError('Expected manifest format he-gene-rl-data-v1')
    try:
        section_spec = spec['sections'][section]
        split_name = section_spec['splits'][part]
    except KeyError as exc:
        raise ValueError(f'Manifest has no section/split {section}/{part}') from exc
    checksums = spec.get('checksums', {})

    def input_path(name):
        if not isinstance(name, str) or not name:
            raise ValueError('Manifest paths must be nonempty strings')
        path = Path(name)
        if not path.is_absolute():
            path = manifest_path.parent / path
        return _checked(path, checksums.get(name), inputs)

    feature_path = input_path(section_spec['features'])
    metadata_path = input_path(section_spec['metadata'])
    split_path = input_path(split_name)
    bank_key = (str(feature_path), inputs[str(feature_path)], inputs[str(metadata_path)])
    if bank_key not in _BANK_CACHE:
        features = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        with np.load(metadata_path, allow_pickle=False) as archive:
            bank_ids = _identities(archive['cell_ids'], 'Bank cell_ids')
            gene_ids = _identities(archive['gene_ids'], 'gene_ids')
            # Keep storage precision for subsequent subtraction/normalization.
            # Only the spatial search itself promotes coordinates to float64.
            coordinates = np.asarray(archive['image_coordinates'])
        if (features.ndim != 2 or features.shape[0] != len(bank_ids)
                or features.shape[1] < 1 or features.dtype.kind not in 'fiu'
                or coordinates.shape != (len(bank_ids), 2) or coordinates.dtype.kind not in 'fiu'
                or not np.isfinite(coordinates).all()):
            raise ValueError('Bank features, coordinates and identities do not align')
        for start in range(0, len(features), 4096):
            if not np.isfinite(features[start:start + 4096]).all():
                raise ValueError('Image bank features must be finite')
        _BANK_CACHE[bank_key] = (features, bank_ids, gene_ids, coordinates, cKDTree(coordinates))
    bank_features, bank_ids, gene_ids, coordinates, tree = _BANK_CACHE[bank_key]
    with np.load(split_path, allow_pickle=False) as archive:
        cell_ids = _identities(archive['cell_ids'], 'Split cell_ids')
        rows = np.asarray(archive['row_indices'])
        if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(rows) != len(cell_ids)
                or len(np.unique(rows)) != len(rows)
                or np.any(rows < 0) or np.any(rows >= len(bank_features))
                or not np.array_equal(bank_ids[rows], cell_ids)):
            raise ValueError('Split row/cell identity join failed')
        if 'gene_ids' in archive.files and not np.array_equal(
                _identities(archive['gene_ids'], 'Split gene_ids'), gene_ids):
            raise ValueError('Split gene identity/order differs from the bank')
        # Historical cohorts may store equal coordinates in a different dtype
        # from their bank. Retain that dtype when explicitly supplied, after
        # verifying the same identity join used by the original loader.
        center_coordinates = (np.asarray(archive['coordinates'])
                              if 'coordinates' in archive.files else coordinates[rows])
        if (center_coordinates.shape != (len(rows), 2)
                or center_coordinates.dtype.kind not in 'fiu'
                or not np.isfinite(center_coordinates).all()
                or not np.array_equal(center_coordinates, coordinates[rows])):
            raise ValueError('Split coordinates differ from the bank identity rows')
        has_tiles = 'tiles' in archive.files
        tiles = np.asarray(archive['tiles']) if has_tiles else np.zeros(len(rows), dtype=np.int64)
        if (tiles.ndim not in (1, 2) or len(tiles) != len(rows)
                or (tiles.ndim == 2 and tiles.shape[1] < 1)
                or tiles.dtype.kind not in 'USfiu'
                or (tiles.dtype.kind in 'fiu' and not np.isfinite(tiles).all())):
            raise ValueError('tiles must provide one finite/string block identity per cell')
        counts = None
        if require_counts:
            if 'counts' not in archive.files:
                raise ValueError(f'Expression targets required for {section}/{part}')
            counts = np.asarray(archive['counts'])
            if (counts.shape != (len(rows), len(gene_ids)) or counts.dtype.kind not in 'fiu'
                    or not np.isfinite(counts).all() or np.any(counts < 0)
                    or np.any(counts != np.floor(counts))):
                raise ValueError('counts must be aligned, finite nonnegative integers')
    original_cells = len(rows)
    if max_cells is not None:
        limit = int(max_cells)
        rows, cell_ids, tiles = rows[:limit], cell_ids[:limit], tiles[:limit]
        center_coordinates = center_coordinates[:limit]
        if counts is not None:
            counts = counts[:limit]
    nearest = neighbor_indices(coordinates, rows, neighbors, tree=tree)
    metadata = {'format': 'he-gene-rl-cohort-v1', 'section': section, 'part': part,
                'inputs': inputs, 'original_cells': original_cells, 'used_cells': len(rows),
                'subset': 'first cells in stored split order' if max_cells is not None else None,
                'neighbor_count': int(neighbors), 'neighbor_expression_used': False,
                'old_model_scores_used': False, 'full_section_image_nodes': len(bank_features),
                'transductive_full_section_image_context': True,
                'spatial_blocks_available': has_tiles, 'expression_loaded': bool(require_counts),
                'coordinate_system': 'bank.image_coordinates; source-train nearest-distance normalized',
                'bank_features': str(feature_path), 'bank_metadata': str(metadata_path)}
    return ContextCohort(section=section, part=part, features=bank_features[rows], counts=counts,
        cell_ids=cell_ids, row_indices=rows, coordinates=center_coordinates, tiles=tiles,
        gene_ids=gene_ids, neighbor_indices=nearest, bank_features=bank_features,
        bank_coordinates=coordinates, metadata=metadata)
