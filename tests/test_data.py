"""Image/RNA isolation, portable identity joins and source normalization.

These tests are supplied for the later validation phase. They were not run
as part of the initial project packaging task.
"""
import json

import numpy as np
import pytest
import torch

from he_gene_rl.data import (
    ContextCohort, fit_normalization, load_cohort, neighbor_indices,
    public_from_features, sha256,
)


def cohort(part='train'):
    positions = np.array([[0., 0.], [1., 0.], [0., 2.], [10., 10.]])
    features = np.array([[1., 3.], [3., 7.], [5., 9.], [100., 90.]], dtype=np.float32)
    rows = np.array([0, 1])
    return ContextCohort('Rep1', part, features[rows].copy(),
        np.array([[0., 2.], [1., 5.]], np.float32), np.array(['a', 'b']), rows,
        positions[rows].copy(), np.array([1, 2]), np.array(['g0', 'g1']),
        neighbor_indices(positions, rows, 2), features, positions, {'inputs': {}})


def write_dataset(root, *, counts=True, tiles=True, rows=None, ids=None, genes=None):
    root.mkdir(exist_ok=True)
    c = cohort()
    bank_ids = np.array(['a', 'b', 'c', 'd'])
    np.save(root / 'features.npy', c.bank_features)
    np.savez(root / 'metadata.npz', cell_ids=bank_ids, gene_ids=c.gene_ids,
             image_coordinates=c.bank_coordinates)
    arrays = {'row_indices': c.row_indices if rows is None else np.asarray(rows),
              'cell_ids': c.cell_ids if ids is None else np.asarray(ids)}
    if counts:
        arrays['counts'] = c.counts
    if tiles:
        arrays['tiles'] = c.tiles
    if genes is not None:
        arrays['gene_ids'] = np.asarray(genes)
    np.savez(root / 'split.npz', **arrays)
    manifest = {'format': 'he-gene-rl-data-v1', 'sections': {
        'sample': {'features': 'features.npy', 'metadata': 'metadata.npz',
                   'splits': {'train': 'split.npz', 'test': 'split.npz', 'predict': 'split.npz'}}}}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


def test_neighbors_exclude_self_and_handle_coincident_positions():
    xy = np.array([[0., 0.], [0., 0.], [1., 0.], [8., 2.]])
    rows = np.arange(4)
    selected = neighbor_indices(xy, rows, 2)
    assert selected.shape == (4, 2)
    assert not np.any(selected == rows[:, None])
    assert selected[0, 0] == 1 and selected[1, 0] == 0
    assert len(np.unique(selected[3])) == 2
    with pytest.raises(ValueError):
        neighbor_indices(xy, np.array([4]), 2)


def test_normalization_uses_source_centers_not_neighbor_bank_or_test():
    c = cohort()
    normal = fit_normalization(c, chunk_size=1)
    np.testing.assert_allclose(normal['feature_mean'], [2., 5.])
    np.testing.assert_allclose(normal['feature_scale'], [1., 2.])
    assert normal['coordinate_scale'] == 1.
    with pytest.raises(ValueError, match='source TRAIN'):
        fit_normalization(cohort('test'))


def test_batch_keeps_neighbor_images_independent_of_rna_targets():
    c = cohort()
    normal = fit_normalization(c)
    first = c.batch(np.array([0, 1]), 'cpu', normal)
    c.counts[:] = 999
    second = c.batch(np.array([0, 1]), 'cpu', normal)
    assert set(first) == {'x', 'neighbors', 'relative_xy', 'neighbor_mask', 'counts'}
    for key in ('x', 'neighbors', 'relative_xy', 'neighbor_mask'):
        torch.testing.assert_close(first[key], second[key])
    assert not torch.equal(first['counts'], second['counts'])
    torch.testing.assert_close(first['relative_xy'][0], torch.tensor([[1., 0.], [0., 2.]]))


def test_target_batch_reuses_source_normalization_without_refitting():
    normal = fit_normalization(cohort())
    target = cohort('test')
    target.features += 20
    x = target.batch(np.array([0]), 'cpu', normal)['x']
    torch.testing.assert_close(x, torch.tensor([[19., 9.]]))
    with pytest.raises(ValueError, match='normalization'):
        target.batch(np.array([0]))


def test_manifest_paths_resolve_from_manifest_not_working_directory(tmp_path, monkeypatch):
    manifest = write_dataset(tmp_path / 'portable')
    monkeypatch.chdir(tmp_path)
    loaded = load_cohort(manifest, 'sample', 'train', neighbors=2)
    expected = cohort()
    np.testing.assert_array_equal(loaded.features, expected.features)
    np.testing.assert_array_equal(loaded.counts, expected.counts)
    np.testing.assert_array_equal(loaded.neighbor_indices, expected.neighbor_indices)
    assert loaded.metadata['spatial_blocks_available'] is True
    assert len(loaded.metadata['inputs']) == 4


def test_unlabeled_prediction_never_loads_even_unreadable_expression_array(tmp_path):
    manifest = write_dataset(tmp_path, counts=False, tiles=False)
    # Object arrays cannot be loaded with allow_pickle=False. The public loader
    # must not touch this key even though its name appears in the archive.
    np.savez(tmp_path / 'split.npz', row_indices=np.array([0, 1]),
             cell_ids=np.array(['a', 'b']), counts=np.array([object()], dtype=object))
    loaded = load_cohort(manifest, 'sample', 'predict', neighbors=2, require_counts=False)
    assert loaded.counts is None
    assert loaded.metadata['expression_loaded'] is False
    assert loaded.metadata['spatial_blocks_available'] is False
    assert len(loaded) == 2 and loaded.gene_count == 2
    public = public_from_features(loaded, np.array([0]), fit_normalization(cohort()), 'cpu')
    assert set(public) == {'x', 'neighbors', 'relative_xy', 'neighbor_mask'}
    with pytest.raises(ValueError, match='unavailable'):
        loaded.batch(np.array([0]), normalization=fit_normalization(cohort()))
    with pytest.raises(ValueError):
        load_cohort(manifest, 'sample', 'train', neighbors=2, require_counts=True)


def test_image_only_schema_does_not_require_target_labels(tmp_path):
    manifest = write_dataset(tmp_path, counts=False)
    loaded = load_cohort(manifest, 'sample', 'predict', neighbors=2, require_counts=False)
    assert loaded.counts is None
    with pytest.raises(ValueError, match='Expression targets required'):
        load_cohort(manifest, 'sample', 'test', neighbors=2)


@pytest.mark.parametrize('rows,ids', [([1, 0], ['a', 'b']), ([0, 0], ['a', 'b']),
                                    ([0, 4], ['a', 'b'])])
def test_loader_rejects_wrong_or_duplicate_cell_join_before_subsetting(tmp_path, rows, ids):
    manifest = write_dataset(tmp_path, rows=rows, ids=ids)
    with pytest.raises(ValueError, match='identity join'):
        load_cohort(manifest, 'sample', 'train', neighbors=2, max_cells=1)


def test_loader_rejects_reordered_gene_columns(tmp_path):
    manifest = write_dataset(tmp_path, genes=['g1', 'g0'])
    with pytest.raises(ValueError, match='gene identity/order'):
        load_cohort(manifest, 'sample', 'train', neighbors=2)


def test_loader_preserves_bank_and_explicit_center_coordinate_precision(tmp_path):
    manifest = write_dataset(tmp_path)
    c = cohort()
    np.savez(tmp_path / 'metadata.npz', cell_ids=np.array(['a', 'b', 'c', 'd']),
             gene_ids=c.gene_ids, image_coordinates=c.bank_coordinates.astype(np.float32))
    np.savez(tmp_path / 'split.npz', row_indices=c.row_indices, cell_ids=c.cell_ids,
             counts=c.counts, coordinates=c.coordinates.astype(np.float64))
    loaded = load_cohort(manifest, 'sample', 'train', neighbors=2)
    assert loaded.bank_coordinates.dtype == np.dtype(np.float32)
    assert loaded.coordinates.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(loaded.coordinates, loaded.bank_coordinates[loaded.row_indices])
    np.savez(tmp_path / 'split.npz', row_indices=c.row_indices, cell_ids=c.cell_ids,
             counts=c.counts, coordinates=c.coordinates + 1.)
    with pytest.raises(ValueError, match='coordinates differ'):
        load_cohort(manifest, 'sample', 'train', neighbors=2)


@pytest.mark.parametrize('counts', [np.array([[0., .5], [1., 2.]]),
                                    np.array([[0., -1.], [1., 2.]]),
                                    np.array([[0., np.nan], [1., 2.]]),
                                    np.zeros((2, 3))])
def test_loader_rejects_invalid_raw_counts(tmp_path, counts):
    manifest = write_dataset(tmp_path)
    np.savez(tmp_path / 'split.npz', row_indices=np.array([0, 1]),
             cell_ids=np.array(['a', 'b']), counts=counts)
    with pytest.raises(ValueError, match='nonnegative integers'):
        load_cohort(manifest, 'sample', 'train', neighbors=2)


def test_optional_manifest_checksums_detect_changed_bank(tmp_path):
    manifest = write_dataset(tmp_path)
    spec = json.loads(manifest.read_text())
    spec['checksums'] = {'features.npy': sha256(tmp_path / 'features.npy')}
    manifest.write_text(json.dumps(spec))
    changed = cohort().bank_features.copy()
    changed[0, 0] += 1
    np.save(tmp_path / 'features.npy', changed)
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_cohort(manifest, 'sample', 'train', neighbors=2)


def test_public_inputs_accept_image_only_duck_type_with_poisoned_counts():
    c = cohort()

    class ImagesOnly:
        features = c.features
        coordinates = c.coordinates
        neighbor_indices = c.neighbor_indices
        bank_features = c.bank_features
        bank_coordinates = c.bank_coordinates

        @property
        def counts(self):
            raise AssertionError('Inference accessed RNA')

    actual = public_from_features(ImagesOnly(), np.array([0, 1]), fit_normalization(c), 'cpu')
    expected = public_from_features(c, np.array([0, 1]), fit_normalization(c), 'cpu')
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
