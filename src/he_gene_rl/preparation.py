"""Rebuild the historical cohorts directly from external Rank H5ad datasets.

Only raw counts, UNI image embeddings, image coordinates, and identities are
read. The spatial boundary and identity-hash recipe reproduce the historical
cohorts without loading old caches, predictions, checkpoints, or Rank modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy import sparse


PARTS = ("train", "policy", "monitor", "calibration", "test")


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _strings(node):
    if isinstance(node, h5py.Dataset):
        values = np.asarray(node[...])
        if values.ndim != 1 or values.dtype.kind not in "OSU":
            raise ValueError(f"Expected a string vector: {node.name}")
        return np.asarray([_text(value) for value in values])
    if _text(node.attrs.get("encoding-type", "")) != "categorical":
        raise ValueError(f"Unsupported string encoding: {node.name}")
    categories = _strings(node["categories"])
    stored_codes = np.asarray(node["codes"][...])
    if stored_codes.ndim != 1 or stored_codes.dtype.kind not in "iu":
        raise ValueError(f"Invalid categorical codes: {node.name}")
    codes = stored_codes.astype(np.int64)
    if codes.ndim != 1 or codes.dtype.kind not in "iu" or np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError(f"Missing or invalid categorical values: {node.name}")
    return categories[codes]


def _counts(values):
    if (values.dtype.kind not in "iuf" or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values != np.floor(values))):
        raise ValueError("layers/raw must contain finite nonnegative integer counts")


def _raw_matrix(node):
    if isinstance(node, h5py.Dataset):
        values = np.asarray(node[...])
    else:
        kind = _text(node.attrs.get("encoding-type", ""))
        constructors = {"csr_matrix": sparse.csr_matrix, "csc_matrix": sparse.csc_matrix}
        if kind not in constructors:
            raise ValueError(f"Unsupported raw matrix encoding: {kind}")
        data, indices, indptr = (np.asarray(node[key][...]) for key in ("data", "indices", "indptr"))
        shape = tuple(int(size) for size in node.attrs["shape"])
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("Invalid sparse count shape")
        major, minor = shape if kind == "csr_matrix" else shape[::-1]
        if (data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1
                or indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu"
                or len(indptr) != major + 1 or len(data) != len(indices)
                or indptr[0] != 0 or indptr[-1] != len(data)
                or np.any(np.less(indptr[1:], indptr[:-1])) or np.any(indices < 0) or np.any(indices >= minor)):
            raise ValueError("Invalid sparse count storage")
        _counts(data)
        values = constructors[kind]((data, indices, indptr), shape=shape).toarray()
    if values.ndim != 2:
        raise ValueError("layers/raw must be a matrix")
    _counts(values)
    result = np.asarray(values, dtype=np.float32)
    if not np.array_equal(values, result):
        raise ValueError("Raw counts cannot be represented exactly in float32")
    return result


def read_section(path):
    """Read only the permitted input fields; never consult prediction layers."""
    with h5py.File(path, "r") as handle:
        required = ("obs", "var/gene_ids", "var/feature_types", "obsm/he", "obsm/image_coor", "layers/raw")
        if any(name not in handle for name in required):
            raise ValueError("H5ad is missing required raw/image/identity fields")
        index_name = _text(handle["obs"].attrs.get("_index", "_index"))
        cell_ids = _strings(handle[f"obs/{index_name}"])
        gene_ids = _strings(handle["var/gene_ids"])
        feature_types = _strings(handle["var/feature_types"])
        if len(feature_types) != len(gene_ids) or not np.all(feature_types == "Gene Expression"):
            raise ValueError("The panel must contain only biological Gene Expression features")
        features = np.asarray(handle["obsm/he"][...], dtype=np.float32)
        coordinates = np.asarray(handle["obsm/image_coor"][...], dtype=np.float64)
        counts = _raw_matrix(handle["layers/raw"])
    for name, identities in (("cell", cell_ids), ("gene", gene_ids)):
        if not len(identities) or len(np.unique(identities)) != len(identities) or np.any(identities == ""):
            raise ValueError(f"Unique nonempty {name} identities are required")
    if (features.ndim != 2 or features.shape[0] != len(cell_ids) or features.shape[1] < 1
            or coordinates.shape != (len(cell_ids), 2)
            or counts.shape != (len(cell_ids), len(gene_ids))
            or not np.isfinite(features).all() or not np.isfinite(coordinates).all()):
        raise ValueError("Image features, counts, coordinates, and identities must align and be finite")
    return {"cell_ids": cell_ids, "gene_ids": gene_ids, "features": features,
            "coordinates": coordinates, "counts": counts}


def _order(rows, cell_ids, seed, namespace):
    return np.asarray(sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}\0{namespace}\0{cell_ids[row]}".encode()).digest()), dtype=np.int64)


def build_splits(cell_ids, coordinates, section, *, seed=42, fit_cells=65536,
                 policy_cells=8192, monitor_cells=1024, calibration_cells=1024,
                 test_cells=4096, previous_held_cells=6144):
    """Historical fixed recipe based exclusively on native IDs and geometry."""
    ids = np.asarray(cell_ids)
    xy: NDArray[np.float64] = np.asarray(coordinates, dtype=np.float64)
    sizes = (fit_cells, policy_cells, monitor_cells, calibration_cells, test_cells)
    if (any(type(size) is not int or size < 1 for size in sizes)
            or type(previous_held_cells) is not int or previous_held_cells < 0
            or type(seed) is not int or seed < 0):
        raise ValueError("Positive integer cohort sizes and nonnegative seed/exclusion size required")
    if (ids.ndim != 1 or ids.dtype.kind not in "US" or len(np.unique(ids)) != len(ids)
            or xy.shape != (len(ids), 2) or not len(ids) or not np.isfinite(xy).all()):
        raise ValueError("Unique identities and aligned finite image coordinates required")
    cutoff = float(np.quantile(xy[:, 0], 0.8))
    original_train = np.flatnonzero(xy[:, 0] < cutoff - 32.0)
    original_validation = np.flatnonzero(xy[:, 0] >= cutoff + 32.0)
    held_size = monitor_cells + calibration_cells + test_cells
    if len(original_train) < fit_cells + policy_cells or len(original_validation) < previous_held_cells + held_size:
        raise ValueError("Insufficient cells for the historical spatial/cohort recipe")
    excluded = _order(original_validation, ids, seed, "sc_gene_rl_held")[:previous_held_cells]
    valid = original_validation[~np.isin(original_validation, excluded)]
    held = _order(valid, ids, seed, "sc_state_transfer_held")
    result = {"train": np.sort(_order(original_train, ids, seed, "sc_state_transfer_fit")[:fit_cells])}
    begin = 0
    for name, size in (("monitor", monitor_cells), ("calibration", calibration_cells), ("test", test_cells)):
        result[name] = np.sort(held[begin:begin + size])
        begin += size
    eligible = np.setdiff1d(original_train, result["train"])
    salt = f"masked_count1_policy:{section}:{seed}"
    policy = sorted(eligible, key=lambda row: hashlib.sha256(f"{salt}:{ids[row]}".encode()).digest())
    result["policy"] = np.sort(np.asarray(policy[:policy_cells], dtype=np.int64))
    return result, {"axis": 0, "validation_fraction": 0.2, "buffer_pixels": 64.0,
                    "cutoff": cutoff, "seed": seed, "previous_held_cells_excluded": len(excluded),
                    "original_train_cells": len(original_train), "original_validation_cells": len(original_validation),
                    "cohort_sizes": {name: len(rows) for name, rows in result.items()},
                    "labels_used_to_partition": False}


def _tiles(xy, edges):
    bins = len(edges) + 1
    return (np.searchsorted(edges[:, 1], xy[:, 1], side="right") * bins
            + np.searchsorted(edges[:, 0], xy[:, 0], side="right")).astype(np.int64)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def prepare(datasets, output, sections=("Rep1", "Rep2"), *, source_variant="plain", **split_options):
    """Create a new portable directory with provenance; source files stay read-only."""
    datasets, output = Path(datasets).resolve(), Path(output).resolve()
    if not sections or len(set(sections)) != len(sections) or any(s not in ("Rep1", "Rep2") for s in sections):
        raise ValueError("Choose unique Rep1/Rep2 sections")
    if source_variant not in ("plain", "with-spatialex"):
        raise ValueError("source_variant must be plain or with-spatialex")
    if output == datasets or datasets in output.parents:
        raise ValueError("Output must be outside the source datasets directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "he-gene-rl-data-v1", "sections": {}, "checksums": {}}
    provenance = {"format": "he-gene-rl-raw-preparation-v1", "sources": {},
                  "source_variant": source_variant,
                  "old_outputs_read": False, "old_modules_imported": False,
                  "prediction_layers_read": False, "source_files_modified": False,
                  "code_sha256": _sha256(__file__),
                  "recipe": "Historical spatial boundary and SHA256 identity namespaces; no stored split files",
                  "image_context": "Full-section UNI/image-coordinate bank; neighbor expression is not an input"}
    expected_genes = None
    for section in sections:
        basename = f"Human_Breast_Cancer_{section}"
        suffix = "_with_spatialex_all_pred" if source_variant == "with-spatialex" else ""
        path = datasets / basename / f"{basename}_uni_resolution64_full{suffix}.h5ad"
        digest = _sha256(path)
        data = read_section(path)
        if expected_genes is not None and not np.array_equal(expected_genes, data["gene_ids"]):
            raise ValueError("Cross-section gene panel or order differs")
        expected_genes = data["gene_ids"]
        rows, recipe = build_splits(data["cell_ids"], data["coordinates"], section, **split_options)
        directory = output / section
        directory.mkdir()
        np.save(directory / "features.npy", data["features"], allow_pickle=False)
        np.savez_compressed(directory / "metadata.npz", cell_ids=data["cell_ids"],
                            gene_ids=data["gene_ids"], image_coordinates=data["coordinates"])
        feature_name, metadata_name = f"{section}/features.npy", f"{section}/metadata.npz"
        split_paths: dict[str, str] = {}
        spec = {"features": feature_name, "metadata": metadata_name, "splits": split_paths}
        edges = {name: np.quantile(data["coordinates"][rows[name]], np.arange(1, 6) / 6, axis=0)
                 for name in ("train", "monitor", "calibration")}
        for part in PARTS:
            selected = rows[part]
            arrays = {"row_indices": selected, "cell_ids": data["cell_ids"][selected],
                      "gene_ids": data["gene_ids"], "coordinates": data["coordinates"][selected],
                      "counts": data["counts"][selected]}
            if part != "policy":
                arrays["tiles"] = _tiles(arrays["coordinates"], edges["calibration" if part == "test" else part])
            np.savez_compressed(directory / f"{part}.npz", **arrays)
            split_paths[part] = f"{section}/{part}.npz"
        if _sha256(path) != digest:
            raise ValueError(f"Source changed while preparing data: {path}")
        manifest["sections"][section] = spec
        for relative in (feature_name, metadata_name, *split_paths.values()):
            manifest["checksums"][relative] = _sha256(output / relative)
        provenance["sources"][section] = {"path": str(path), "sha256": digest, "split_recipe": recipe,
            "n_cells": len(data["cell_ids"]), "n_genes": len(data["gene_ids"]),
            "feature_dim": data["features"].shape[1],
            "fields_read": ["obs index", "var/gene_ids", "var/feature_types", "obsm/he", "obsm/image_coor", "layers/raw"]}
        print(f"Prepared {section}: {recipe['cohort_sizes']}", flush=True)
    _json(output / "preparation_provenance.json", provenance)
    _json(output / "manifest.json", manifest)
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", "--datasets", dest="datasets", type=Path,
                        required=True, help="External Rank/datasets directory")
    parser.add_argument("--output", type=Path, required=True, help="New portable data directory")
    parser.add_argument("--sections", nargs="+", choices=("Rep1", "Rep2"), default=["Rep1", "Rep2"])
    parser.add_argument("--source-variant", choices=("plain", "with-spatialex"), default="plain",
                        help="plain: *_full.h5ad; with-spatialex: historical file variant, prediction layers still ignored")
    args = parser.parse_args()
    print(prepare(args.datasets, args.output, args.sections, source_variant=args.source_variant), flush=True)


if __name__ == "__main__":
    main()
