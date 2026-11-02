"""Score completed gene selections and compare aligned per-cell outcomes."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import final_metrics
from .data import load_cohort, sha256
from .information import paired_block_summary
from .io import assert_frozen, component, reject_existing
from .pretraining import write_json
from .workflow import METHODS, assert_heldout_cells, predict


def aligned_selection(path, cohort):
    """Require the same cells and ordered gene panel; align cell ordering."""
    with np.load(path, allow_pickle=False) as saved:
        key = "selected_mask" if "selected_mask" in saved else "selected"
        selected = saved[key]
        cell_ids, gene_ids = saved["cell_ids"].astype(str), saved["gene_ids"].astype(str)
        sequence = saved["sequence"] if "sequence" in saved else None
    if (selected.dtype != bool or selected.shape != (len(cell_ids), len(gene_ids))
            or cell_ids.ndim != 1 or gene_ids.ndim != 1):
        raise ValueError("Expected boolean [cells, genes] selections with explicit IDs")
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("Duplicate prediction cell IDs")
    if not np.array_equal(gene_ids, cohort.gene_ids.astype(str)):
        raise ValueError("Prediction and truth gene panel/order differ")
    wanted = cohort.cell_ids.astype(str)
    if set(cell_ids) != set(wanted):
        raise ValueError("Prediction and evaluation cell sets differ")
    if sequence is not None:
        if (sequence.shape != selected.shape or sequence.dtype.kind not in "iu"
                or np.any(sequence < -1) or np.any(sequence >= len(gene_ids))):
            raise ValueError("Invalid padded gene selection sequence")
        valid = sequence >= 0
        if np.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("Sequence padding must follow all actions")
        recorded = np.zeros_like(selected, dtype=np.int64)
        rows, columns = np.nonzero(valid)
        np.add.at(recorded, (rows, sequence[rows, columns]), 1)
        if np.any(recorded > 1) or not np.array_equal(recorded > 0, selected):
            raise ValueError("Sequence repeats genes or differs from final selection")
    lookup = {value: index for index, value in enumerate(cell_ids)}
    return selected[np.asarray([lookup[value] for value in wanted])]


def score(path, cohort):
    selected = aligned_selection(path, cohort)
    if cohort.counts is None:
        raise ValueError("Evaluation requires held-out counts")
    return {k.removeprefix("cell_"): v for k, v in final_metrics(cohort.counts, selected).items()}


def mean_metrics(metrics):
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def comparison(candidate, reference, cohort, bootstrap_replicates=2000):
    differences = {key: candidate[key] - reference[key] for key in ("fp", "fn", "cost")}
    if cohort.metadata.get("spatial_blocks_available", False):
        result = paired_block_summary(differences, cohort.tiles, bootstrap_replicates, seed=80421)
    else:
        result = {"n_cells": len(cohort), "n_blocks": None, "bootstrap_replicates": 0,
                  "direction": "candidate minus reference; negative means lower cost/error",
                  "uncertainty_note": "No spatial blocks supplied; no confidence interval computed"}
        result.update({key: {"mean": float(value.mean()), "ci95": None,
                             "n_valid_cells": len(value)} for key, value in differences.items()})
    reference_cost = float(reference["cost"].mean())
    result["relative_cost_reduction_percent"] = (
        100 * (reference_cost - float(candidate["cost"].mean())) / reference_cost
        if reference_cost > 0 else None)
    return result


def evaluate(predictions, manifest, target, split, output, neighbors=8):
    reject_existing(output)
    cohort = load_cohort(manifest, target, split, neighbors=neighbors, require_counts=True)
    metrics = score(predictions, cohort)
    report = {"target": target, "split": split, "n_cells": len(cohort),
              "gene_count": cohort.gene_count, "means": mean_metrics(metrics),
              "cost": "FP + 2*FN; raw0 negative, raw1 neutral, raw>=2 positive",
              "predictions_sha256": sha256(predictions)}
    write_json(output, report)
    return report


def compare(candidate, reference, manifest, target, split, output,
            neighbors=8, bootstrap_replicates=2000):
    reject_existing(output)
    cohort = load_cohort(manifest, target, split, neighbors=neighbors, require_counts=True)
    proposed, baseline = score(candidate, cohort), score(reference, cohort)
    report = {"target": target, "split": split, "candidate": mean_metrics(proposed),
              "reference": mean_metrics(baseline),
              "paired_difference": comparison(proposed, baseline, cohort, bootstrap_replicates),
              "input_sha256": {"candidate": sha256(candidate), "reference": sha256(reference)},
              "uncertainty_scope": "spatial blocks within these cells, not patient generalization"}
    write_json(output, report)
    return report


def evaluate_run(run, manifest, output, device="cpu", methods=None, split="test"):
    """Freeze all models, create predictions, then score with held-out counts."""
    methods = tuple(methods or METHODS)
    if len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Unknown or duplicate evaluation methods")
    frozen = assert_frozen(run)
    config = frozen["config"]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserving an existing evaluation; use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    settings = config.get("evaluation", {})
    targets = settings.get("targets", config["sources"])
    batch_size = int(settings.get("batch_size", 512))
    bootstrap = int(settings.get("bootstrap_replicates", 2000))
    # All prediction commands load image-only cohorts. No test labels are opened
    # until every requested source/target/seed/method has completed its trajectory.
    paths = {}
    for source in config["sources"]:
        for target in targets:
            component(target)
            direction = f"{source}_to_{target}"
            images = load_cohort(manifest, target, split, neighbors=config["neighbors"], require_counts=False)
            assert_heldout_cells(run, source, target, images.cell_ids)
            del images
            for seed in config["seeds"]:
                for method in methods:
                    key = f"{direction}_seed{seed}_{method}"
                    path = output / "predictions" / f"{key}.npz"
                    predict(run, manifest, source, target, seed, split, path, device, method, batch_size)
                    paths[key] = path
    assert_frozen(run)
    directions = {}
    for source in config["sources"]:
        for target in targets:
            direction = f"{source}_to_{target}"
            cohort = load_cohort(manifest, target, split, neighbors=config["neighbors"], require_counts=True)
            per_seed, averaged = {}, {}
            for method in methods:
                records = []
                for seed in config["seeds"]:
                    record = score(paths[f"{direction}_seed{seed}_{method}"], cohort)
                    per_seed[f"{method}/seed{seed}"] = mean_metrics(record)
                    records.append(record)
                averaged[method] = {k: np.stack([r[k] for r in records]).mean(0) for k in records[0]}
            comparisons = {}
            if "ppo_sequence" in methods:
                for reference in methods:
                    if reference != "ppo_sequence":
                        comparisons[f"ppo_sequence_vs_{reference}"] = comparison(
                            averaged["ppo_sequence"], averaged[reference], cohort, bootstrap)
            directions[direction] = {"n_cells": len(cohort), "gene_count": cohort.gene_count,
                "n_seeds": len(config["seeds"]), "cross_section": source != target,
                "means": {m: mean_metrics(v) for m, v in averaged.items()},
                "per_seed": per_seed, "comparisons": comparisons}
    report = {"format": "he-gene-rl-evaluation-v1", "directions": directions,
              "methods": list(methods), "split": split,
              "uncertainty": "average seeds per cell, then paired spatial-block bootstrap",
              "all_predictions_finished_before_test_labels": True,
              "prediction_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in paths.values()}}
    write_json(output / "evaluation_summary.json", report)
    lines = ["HE-Gene-RL evaluation", "Cost: FP+2FN (lower is better); raw1 is neutral.", ""]
    for direction, values in directions.items():
        lines.extend([direction, "method | FP | FN | cost | selected"])
        for method, m in values["means"].items():
            lines.append(f"{method} | {m['fp']:.6f} | {m['fn']:.6f} | {m['cost']:.6f} | {m['selected']:.6f}")
        lines.append("")
    (output / "RESULTS.txt").write_text("\n".join(lines), encoding="utf-8")
    return report
