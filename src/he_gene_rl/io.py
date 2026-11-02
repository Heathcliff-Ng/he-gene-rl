"""Portable run manifests and artifact integrity helpers."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .data import sha256
from .pretraining import write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (".", ".."):
        raise ValueError(f"Expected a simple section/method name: {value!r}")
    return value


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def source_fingerprint():
    root = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(root.glob("*.py"))}


def bind_run(output, config, manifest):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = {"format": "he-gene-rl-run-v1", "origin": "standalone-training",
              "config": config, "data_manifest_sha256": sha256(manifest),
              "code": source_fingerprint(), "read_only": False}
    path = output / "run.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("read_only"):
            raise ValueError("Imported historical runs are inference-only; use a new training output")
        if previous != record:
            raise ValueError("Run code/config/data manifest changed; choose a new output directory")
    elif any(output.iterdir()):
        raise ValueError("Training output is nonempty but has no matching run.json")
    else:
        write_json(path, record)
    return record


def bind_data_inputs(output, manifest, inputs):
    root = Path(manifest).resolve().parent
    portable = {}
    for name, digest in inputs.items():
        path = Path(name).resolve()
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            key = str(path)
        portable[key] = digest
    path = Path(output) / "training_inputs.json"
    if path.exists() and read_json(path) != portable:
        raise ValueError("Training data content changed since this run began")
    write_json(path, portable)


def freeze_run(output, config):
    output = Path(output)
    paths = [output / "run.json", output / "training_summary.json", output / "training_inputs.json"]
    for source in config["sources"]:
        directory = output / component(source)
        paths.extend(directory / name for name in ("normalization.npz", "genes.json", "split_ids.npz"))
        if (directory / "preprocessing.json").exists():
            paths.append(directory / "preprocessing.json")
        for seed in config["seeds"]:
            for stage in ("pretraining", "adaptation", "ppo"):
                paths.append(directory / f"seed{seed}" / stage / "best.pt")
    artifacts = {p.relative_to(output).as_posix(): sha256(p) for p in paths}
    record = {"format": "he-gene-rl-freeze-v1", "origin": "standalone-training",
              "config": config, "artifacts": artifacts, "complete": True,
              "test_labels_used_during_training": False}
    write_json(output / "training_frozen.json", record)
    return record


def assert_frozen(output):
    output = Path(output).resolve()
    record = read_json(output / "training_frozen.json")
    if record.get("format") != "he-gene-rl-freeze-v1" or not record.get("complete"):
        raise ValueError("A complete frozen run is required")
    artifacts = record.get("artifacts", {})
    required = {"run.json"}
    for source in record["config"]["sources"]:
        source = component(source)
        required.update(f"{source}/{name}" for name in ("normalization.npz", "genes.json", "split_ids.npz"))
        for seed in record["config"]["seeds"]:
            required.update(f"{source}/seed{seed}/{stage}/best.pt"
                            for stage in ("pretraining", "adaptation", "ppo"))
    if not required.issubset(artifacts):
        raise ValueError("Frozen run is missing required model/data identities")
    for relative, digest in artifacts.items():
        path = (output / relative).resolve()
        if output not in path.parents or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed frozen artifact: {relative}")
    run = read_json(output / "run.json")
    if run.get("config") != record["config"]:
        raise ValueError("Frozen configuration differs from the checked run manifest")
    if record.get("origin") == "standalone-training":
        if run.get("code") != source_fingerprint():
            raise ValueError("Standalone run configuration or implementation differs from its training freeze")
    return record


def reject_existing(path):
    if Path(path).exists():
        raise FileExistsError(f"Preserving existing output; choose a new path: {path}")
