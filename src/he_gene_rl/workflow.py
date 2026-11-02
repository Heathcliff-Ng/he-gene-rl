"""Train all three stages, or deploy a frozen predictor and PPO policy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .adaptation import fit_adapter
from .core import rollout
from .data import fit_normalization, load_cohort, public_from_features, sha256
from .io import (assert_frozen, bind_data_inputs, bind_run, component, freeze_run,
                 read_json, reject_existing, save_npz)
from .model import ContextPredictor
from .ppo import fit_policy, load_policy, predictor_fingerprint
from .pretraining import fit, load_model, write_json


METHODS = ("teacher_sequence", "adapted_sequence", "ppo_sequence",
           "teacher_one_shot", "adapted_one_shot")


def assert_heldout_cells(run, source, target, cell_ids):
    if source != target:
        return
    with np.load(Path(run) / source / "split_ids.npz", allow_pickle=False) as fitted:
        fitted_ids = set().union(*(set(fitted[k].astype(str)) for k in fitted.files))
    if fitted_ids & set(np.asarray(cell_ids).astype(str)):
        raise ValueError("Test cells overlap fitted source cells")


def runtime(device):
    if str(device) != "cpu" and not str(device).startswith("cuda"):
        raise ValueError("Supported devices are cpu, cuda, or cuda:N")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; specify --device cpu")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False


def validate_config(config):
    for key in ("sources", "seeds", "neighbors", "hidden_dim", "pretraining", "adaptation", "ppo"):
        if key not in config:
            raise ValueError(f"Missing configuration field: {key}")
    if not config["sources"] or len(set(config["sources"])) != len(config["sources"]):
        raise ValueError("Unique, nonempty sources are required")
    for source in config["sources"]:
        component(source)
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Unique nonnegative integer seeds are required")
    for key in ("neighbors", "hidden_dim"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")


def _check_training_parts(parts, config):
    train = parts["train"]
    for name, cohort in parts.items():
        if not np.array_equal(cohort.gene_ids.astype(str), train.gene_ids.astype(str)):
            raise ValueError(f"Gene order differs in {name}")
        if cohort.feature_dim != train.feature_dim:
            raise ValueError("Feature dimensions differ between splits")
    for first, second in (("train", "policy"), ("train", "monitor"), ("policy", "monitor")):
        if set(parts[first].cell_ids.astype(str)) & set(parts[second].cell_ids.astype(str)):
            raise ValueError(f"Overlapping {first}/{second} cells")
    for stage in ("pretraining", "adaptation"):
        budgets = config[stage]["history_budgets"]
        if not budgets or any(type(b) is not int or not 0 <= b <= train.gene_count for b in budgets):
            raise ValueError(f"Invalid {stage} history budgets for this gene panel")
    if train.gene_count < 32:
        raise ValueError("This faithful pretraining implementation requires at least 32 genes")


def train(config_path, manifest, output, stage="all", device=None):
    config = read_json(config_path)
    if device is not None:
        config["device"] = device
    validate_config(config)
    device = config.get("device", "cpu")
    runtime(device)
    if stage not in ("all", "pretrain", "adapt", "ppo"):
        raise ValueError("Unknown training stage")
    output = Path(output).resolve()
    bind_run(output, config, manifest)
    if (output / "training_frozen.json").exists():
        assert_frozen(output)
        print("Run already frozen; preserving completed training.", flush=True)
        return

    cohorts, inputs = {}, {}
    for source in config["sources"]:
        parts = {part: load_cohort(manifest, source, part, neighbors=config["neighbors"],
                                 max_cells=config.get("limits", {}).get(part), require_counts=True)
                 for part in ("train", "policy", "monitor")}
        _check_training_parts(parts, config)
        cohorts[source] = parts
        for cohort in parts.values():
            inputs.update(cohort.metadata["inputs"])
    first_source = cohorts[config["sources"][0]]["train"]
    for source, parts in cohorts.items():
        if not np.array_equal(parts["train"].gene_ids.astype(str), first_source.gene_ids.astype(str)):
            raise ValueError(f"Cross-section gene panel/order differs in {source}")
    bind_data_inputs(output, manifest, inputs)
    summary_path = output / "training_summary.json"
    summaries = read_json(summary_path) if summary_path.exists() else {}
    for source, parts in cohorts.items():
        source_dir = output / source
        source_dir.mkdir(parents=True, exist_ok=True)
        norm_path = source_dir / "normalization.npz"
        identity_path = source_dir / "preprocessing.json"
        if identity_path.exists():
            for name, digest in read_json(identity_path).items():
                if sha256(source_dir / name) != digest:
                    raise ValueError(f"Source preprocessing changed during resume: {source}/{name}")
            with np.load(norm_path, allow_pickle=False) as saved:
                normalization = {k: saved[k] for k in saved.files}
        else:
            normalization = fit_normalization(parts["train"])
            save_npz(norm_path, **normalization)
            write_json(source_dir / "genes.json", parts["train"].gene_ids.astype(str).tolist())
            save_npz(source_dir / "split_ids.npz", **{k: v.cell_ids.astype(str) for k, v in parts.items()})
            write_json(identity_path, {name: sha256(source_dir / name)
                       for name in ("normalization.npz", "genes.json", "split_ids.npz")})
        for seed in config["seeds"]:
            run_dir = source_dir / f"seed{seed}"
            teacher = run_dir / "pretraining" / "best.pt"
            key = f"{source}/seed{seed}"
            record = summaries.setdefault(key, {})
            if stage in ("all", "pretrain"):
                torch.manual_seed(seed)
                if str(device).startswith("cuda"):
                    torch.cuda.manual_seed_all(seed)
                model = ContextPredictor(parts["train"].feature_dim, parts["train"].gene_count,
                                         config["hidden_dim"], use_neighbors=True).to(device)
                model.fit_count_statistics(parts["train"].counts)
                model, record["pretraining"] = fit(model, parts["train"], parts["monitor"],
                    normalization, config["pretraining"], run_dir / "pretraining", seed, device)
                del model
                write_json(summary_path, summaries)
            if stage in ("all", "adapt"):
                if not teacher.is_file():
                    raise FileNotFoundError("Run --stage pretrain first")
                model, record["adaptation"] = fit_adapter(teacher, parts["train"], parts["monitor"],
                    normalization, config["adaptation"], run_dir / "adaptation", seed, device)
                del model
                write_json(summary_path, summaries)
            if stage in ("all", "ppo"):
                adapter = run_dir / "adaptation" / "best.pt"
                if not adapter.is_file():
                    raise FileNotFoundError("Run --stage adapt first")
                model, _ = load_model(adapter, device)
                actor, record["ppo"] = fit_policy(model, parts["policy"], parts["monitor"],
                    normalization, config["ppo"], run_dir / "ppo", seed, device)
                del model, actor
                write_json(summary_path, summaries)
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed during training: {path}")
    if all((output / source / f"seed{seed}" / phase / "summary.json").exists()
           for source in config["sources"] for seed in config["seeds"]
           for phase in ("pretraining", "adaptation", "ppo")):
        freeze_run(output, config)
    print(f"Completed stage {stage}: {output}", flush=True)


@torch.no_grad()
def one_shot(model, cohort, normalization, batch_size, device):
    model.eval()
    selections, point_values = [], []
    for start in range(0, len(cohort.features), batch_size):
        indices = np.arange(start, min(start + batch_size, len(cohort.features)))
        public = public_from_features(cohort, indices, normalization, device)
        mask = torch.zeros((len(indices), model.gene_count), dtype=torch.bool, device=device)
        output = model(public, torch.zeros_like(mask, dtype=public["x"].dtype), mask)
        selections.append((output["utility"] > 0).cpu().numpy())
        point_values.append(output["log_count"].expm1().cpu().numpy())
    return {"selected_mask": np.concatenate(selections),
            "predicted_counts": np.concatenate(point_values)}


def predict(run, manifest, source, target, seed, split, output, device="cpu",
            method="ppo_sequence", batch_size=512):
    runtime(device)
    if method not in METHODS or batch_size < 1:
        raise ValueError("Unknown method or invalid batch size")
    source, target = component(source), component(target)
    run, output = Path(run).resolve(), Path(output)
    reject_existing(output)
    sidecar = output.with_suffix(output.suffix + ".json")
    reject_existing(sidecar)
    frozen = assert_frozen(run)
    config = frozen["config"]
    if source not in config["sources"] or seed not in config["seeds"]:
        raise ValueError("Source/seed is not part of this frozen run")
    cohort = load_cohort(manifest, target, split, neighbors=config["neighbors"], require_counts=False)
    genes = np.asarray(read_json(run / source / "genes.json"), dtype=str)
    if not np.array_equal(genes, cohort.gene_ids.astype(str)):
        raise ValueError("Prediction gene panel/order differs from the trained model")
    if source == target and split == "test":
        assert_heldout_cells(run, source, target, cohort.cell_ids)
    with np.load(run / source / "normalization.npz", allow_pickle=False) as saved:
        normalization = {k: saved[k] for k in saved.files}
    directory = run / source / f"seed{seed}"
    phase = "pretraining" if method.startswith("teacher_") else "adaptation"
    model, _ = load_model(directory / phase / "best.pt", device)
    if model.feature_dim != cohort.feature_dim or model.gene_count != cohort.gene_count:
        raise ValueError("Checkpoint and prediction inputs have incompatible dimensions")
    actor = None
    if method == "ppo_sequence":
        actor, payload = load_policy(directory / "ppo" / "best.pt", device)
        if payload.get("predictor_sha256") != predictor_fingerprint(model):
            raise ValueError("PPO checkpoint belongs to a different predictor")
    prediction = (one_shot(model, cohort, normalization, batch_size, device)
                  if method.endswith("one_shot")
                  else rollout(model, cohort, normalization, actor, batch_size, device))
    arrays = {**prediction, "cell_ids": cohort.cell_ids.astype(str), "gene_ids": genes}
    if cohort.metadata.get("spatial_blocks_available", False):
        arrays["tiles"] = cohort.tiles
    save_npz(output, **arrays)
    write_json(sidecar, {"format": "he-gene-rl-prediction-v1", "source": source, "target": target,
        "seed": seed, "split": split, "method": method, "n_cells": len(cohort),
        "gene_count": cohort.gene_count, "test_expression_read": False,
        "prediction_sha256": sha256(output), "predictor_sha256": sha256(directory / phase / "best.pt"),
        "policy_sha256": sha256(directory / "ppo" / "best.pt") if actor is not None else None})
    print(f"Saved predictions: {output}", flush=True)
    return prediction
