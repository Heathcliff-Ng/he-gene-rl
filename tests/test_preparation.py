"""Direct H5ad preparation preserves raw labels and independent spatial splits."""
import hashlib
import json

import h5py
import numpy as np
import pytest
from scipy import sparse

from he_gene_rl.data import load_cohort
from he_gene_rl.preparation import build_splits, prepare, read_section


SIZES = dict(fit_cells=40, policy_cells=10, monitor_cells=4, calibration_cells=4,
             test_cells=8, previous_held_cells=6)


def write_h5ad(root, storage="dense"):
    directory = root / "Human_Breast_Cancer_Rep1"
    directory.mkdir(parents=True)
    path = directory / "Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad"
    n, genes = 200, 32
    counts = np.arange(n * genes, dtype=np.float32).reshape(n, genes) % 5
    features = np.arange(n * 4, dtype=np.float32).reshape(n, 4) / 10
    xy = np.column_stack((np.arange(n) * 100, np.arange(n) % 13)).astype(np.float64)
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "cell_id"
        obs.create_dataset("cell_id", data=np.asarray([f"c{i}" for i in range(n)], dtype="S"))
        var = handle.create_group("var")
        var.create_dataset("gene_ids", data=np.asarray([f"g{i}" for i in range(genes)], dtype="S"))
        category = var.create_group("feature_types")
        category.attrs["encoding-type"] = "categorical"
        category.create_dataset("categories", data=np.asarray(["Gene Expression"], dtype="S"))
        category.create_dataset("codes", data=np.zeros(genes, dtype=np.int8))
        handle.create_dataset("obsm/he", data=features)
        handle.create_dataset("obsm/image_coor", data=xy)
        if storage == "dense":
            handle.create_dataset("layers/raw", data=counts)
        else:
            matrix = sparse.csr_matrix(counts) if storage == "csr_matrix" else sparse.csc_matrix(counts)
            node = handle.create_group("layers/raw")
            node.attrs["encoding-type"] = storage
            node.attrs["shape"] = matrix.shape
            for name in ("data", "indices", "indptr"):
                node.create_dataset(name, data=getattr(matrix, name))
        # Invalid normalized/prediction values must never be used as targets.
        handle.create_dataset("X", data=np.full((n, genes), np.nan))
        handle.create_dataset("layers/spatialex_a_model_pred", data=np.full((n, genes), np.inf))
    return path, counts, features, xy


@pytest.mark.parametrize("storage", ["dense", "csr_matrix", "csc_matrix"])
def test_reads_raw_labels_and_images_without_prediction_layers(tmp_path, storage):
    path, counts, features, xy = write_h5ad(tmp_path, storage)
    result = read_section(path)
    np.testing.assert_array_equal(result["counts"], counts)
    np.testing.assert_array_equal(result["features"], features)
    np.testing.assert_array_equal(result["coordinates"], xy)
    assert result["coordinates"].dtype == np.float64


def test_spatial_split_disjoint_and_reproduces_historical_held_exclusion(tmp_path):
    path, _, _, xy = write_h5ad(tmp_path)
    ids = read_section(path)["cell_ids"]
    parts, audit = build_splits(ids, xy, "Rep1", **SIZES)
    assert len(np.unique(np.concatenate(list(parts.values())))) == sum(map(len, parts.values()))
    assert np.max(xy[np.concatenate((parts["train"], parts["policy"]))], axis=0)[0] < audit["cutoff"] - 32
    held = np.concatenate([parts[name] for name in ("monitor", "calibration", "test")])
    assert np.min(xy[held, 0]) >= audit["cutoff"] + 32
    original = np.flatnonzero(xy[:, 0] >= audit["cutoff"] + 32)
    excluded = sorted(original, key=lambda row: hashlib.sha256(f"42\0sc_gene_rl_held\0{ids[row]}".encode()).digest())[:6]
    assert not np.intersect1d(held, excluded).size
    repeat, _ = build_splits(ids, xy, "Rep1", **SIZES)
    for name, rows in parts.items():
        np.testing.assert_array_equal(rows, repeat[name])
    with pytest.raises(ValueError, match="Insufficient"):
        build_splits(ids, xy, "Rep1")


def test_portable_export_identity_checksums_and_source_preservation(tmp_path):
    source = tmp_path / "datasets"
    path, counts, features, xy = write_h5ad(source)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "prepared"
    manifest = prepare(source, output, sections=["Rep1"], **SIZES)
    train = load_cohort(manifest, "Rep1", "train", neighbors=2)
    test = load_cohort(manifest, "Rep1", "test", neighbors=2, require_counts=False)
    np.testing.assert_array_equal(train.counts, counts[train.row_indices])
    np.testing.assert_array_equal(train.features, features[train.row_indices])
    np.testing.assert_array_equal(train.coordinates, xy[train.row_indices])
    assert test.counts is None
    assert not set(train.cell_ids) & set(test.cell_ids)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    provenance = json.loads((output / "preparation_provenance.json").read_text())
    assert provenance["old_outputs_read"] is False
    assert provenance["prediction_layers_read"] is False
    with pytest.raises(FileExistsError):
        prepare(source, output, sections=["Rep1"], **SIZES)
    with pytest.raises(ValueError, match="outside"):
        prepare(source, source / "prepared", sections=["Rep1"], **SIZES)
    with h5py.File(path, "r+") as handle:
        handle["layers/raw"][0, 0] = 0.25
    with pytest.raises(ValueError, match="integer counts"):
        read_section(path)
