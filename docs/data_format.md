# Portable data format

The package consumes precomputed image embeddings and cell coordinates. It does not extract features from H&E images or require an existing Rank Python installation. The example manifest is [examples/data_manifest.json](../examples/data_manifest.json).

## Directory and manifest

A dataset can be arranged as follows:

```text
data/
  manifest.json
  Rep1/
    features.npy
    metadata.npz
    train.npz
    policy.npz
    monitor.npz
    calibration.npz
    test.npz
  Rep2/
    ...
```

```json
{
  "format": "he-gene-rl-data-v1",
  "sections": {
    "Rep1": {
      "features": "Rep1/features.npy",
      "metadata": "Rep1/metadata.npz",
      "splits": {
        "train": "Rep1/train.npz",
        "policy": "Rep1/policy.npz",
        "monitor": "Rep1/monitor.npz",
        "calibration": "Rep1/calibration.npz",
        "test": "Rep1/test.npz"
      }
    }
  }
}
```

Section names must match the configuration and CLI arguments. Paths are resolved relative to the manifest file; avoid machine-specific absolute paths when sharing a dataset. The normal loader does not resolve paths through the original workspace.

## Image bank

Let `B` be the number of bank cells, `D` the image-feature dimension, and `G` the gene-panel size.

| File / field | Shape | Meaning |
|---|---|---|
| `features.npy` | `[B, D]` | Finite numeric image embeddings, one row per cell |
| `metadata.npz: cell_ids` | `[B]` | Unique string cell identities in bank order |
| `metadata.npz: image_coordinates` | `[B, 2]` | Finite numeric positions in one consistent image coordinate system |
| `metadata.npz: gene_ids` | `[G]` | Unique string gene identities in fixed output/target order |

Use non-object NumPy arrays so archives can be loaded with `allow_pickle=False`. The historical bank used `D = 1024` and `G = 313`. All sections evaluated with the same checkpoint must have matching feature dimensions and the exact same gene order. Do not silently reorder the target counts or output heads.

The retained pretraining monitor uses history budgets up to 32, so the current training workflow requires at least 32 genes. Both configured training-history budget lists must also fit within the panel; their defaults include 64. A smaller panel therefore requires corresponding configuration changes.

The bank contains image features and coordinates, not neighbor expression. A split's center features are indexed from the bank rather than duplicated in every split file.

## Split archives

Let `N` be the number of cells in a split.

| Field | Shape | Required for |
|---|---|---|
| `cell_ids` | `[N]` | All uses; unique cell identities |
| `row_indices` | `[N]` | All uses; integer bank row for each cell |
| `coordinates` | `[N, 2]` | Optional numeric center positions; values must equal the corresponding bank rows; original dtype is preserved |
| `counts` | `[N, G]` | Training, monitoring, and evaluation; not prediction |
| `tiles` | `[N]` or `[N, K]`, `K >= 1` | Optional spatial block identities for uncertainty estimation |

Every `cell_ids[i]` must match the bank identity at `row_indices[i]`. Row indices must be unique within a split and within the bank's bounds. Counts must be finite nonnegative integers in the manifest's gene order; log-normalized or scaled expression is not a substitute for these raw targets.

Without `coordinates`, center positions are indexed from the bank. The legacy
exporter includes this small array to preserve the original center-coordinate
precision even when it differed from the bank's storage dtype. Spatial search
uses float64 internally, while stored coordinates retain their original dtype
for subtraction and source-normalized relative positions.

Cell/gene identities must be nonempty unique Unicode or byte strings. A split may additionally include `gene_ids`; when present, they must exactly match the bank's ordered identities. Block identities may be numeric or strings; numeric block values must be finite. Counts may have an integer or floating dtype, but every value must be integer-valued.

The historical outcome convention is `count = 0` negative, `count >= 2` positive, and `count = 1` neutral. Preserve raw ones; converting them to zeros changes both the training objective and final evaluation.

`train`, `policy`, and `monitor` serve different roles:

- `train`: both supervised predictor stages and source normalization.
- `policy`: PPO trajectory sampling and rewards.
- `monitor`: source-only checkpoint selection.
- `calibration`: optional partition retained for baseline threshold calibration or later comparisons.
- `test`: final evaluation after model selection.

Keep fitting, monitor, and evaluation cell identities disjoint within a section. When preparing a new dataset, choose these partitions before tuning and retain their identities. The package's loader validates each archive; the training/evaluation workflow additionally enforces the relevant split separation.

## Unlabeled deployment

A prediction-only split can contain just `cell_ids` and `row_indices`. Give it a name such as `predict` in the manifest:

```json
"splits": {"predict": "Rep2/predict.npz"}
```

Pass `--split predict` to `he-gene-rl predict`. The prediction path uses `require_counts=False`, so it does not read count targets even if a split file happens to include them. Its public tensors contain only center images, neighbor images, relative positions, and a neighbor-validity mask. Predicted count history is generated by the model.

Evaluation is a separate command and requires the aligned count targets. The prediction archive retains cell and gene identities so an evaluator can check correspondence before computing metrics.

## Prediction and external-baseline archives

Both `evaluate` and `compare` accept an NPZ with these arrays:

| Field | Shape | Requirement |
|---|---|---|
| `selected_mask`, or `selected` | `[N, G]` | Boolean final selection for every cell and every gene |
| `cell_ids` | `[N]` | Unique cell IDs covering exactly the evaluation split |
| `gene_ids` | `[G]` | The complete panel in the exact dataset order |
| `sequence` | `[N, G]` | Optional integer selection order, right-padded with `-1` |

An external baseline need not emit a sequence or count predictions. A boolean selection mask is sufficient. Convert score outputs into decisions with the baseline's predetermined rule or a source-only calibrated threshold before creating its archive. Do not choose thresholds using the evaluation labels.

The evaluator aligns prediction rows to dataset rows by `cell_ids`; a different cell order is allowed. Missing/extra/duplicate cells or a different gene panel/order are rejected. Gene columns are not silently reordered. A top-k list alone is insufficient: construct the full boolean panel mask first.

If an archive contains `sequence`, its nonnegative entries must be unique valid gene indices, occur before all padding, and reproduce the final selection mask. Omit that field for an external method unless it represents actual committed selections in this format.

The package's sequence predictions additionally contain `predicted_values` and `stop_steps`; one-shot predictions instead contain `predicted_counts` for the full panel. Prediction metadata is saved alongside the NPZ as a `.npz.json` file. Evaluation uses the explicit identities and the final selection; it does not use predicted count values as true expression.

## Neighbors and normalization

Spatial neighbors are obtained from the image bank with a nearest-neighbor query that excludes the center cell itself. The bank must contain more cells than the requested neighbor count. Relative coordinates are calculated from bank positions; the caller supplies source-training normalization for all splits.

Feature means/scales and a nearest-distance coordinate scale are fitted on the source `train` cohort only and stored with the run. Prediction on another section must reuse those values. Do not fit a new normalization to the target section.

The historical experiment uses a full-section image bank, including image nodes outside the center's training split. A new dataset may use a different image-context access convention, but that changes the experimental setup and should be reported.

## Spatial blocks

Spatial blocks group nearby cells for uncertainty estimates. Supply stable block identities in `tiles` if these intervals are needed. Missing blocks do not prevent point estimates or unlabeled prediction. The loader records that blocks are unavailable rather than treating a placeholder as genuine spatial replication. A single block cannot support a meaningful spatial-block confidence interval.

## Migrating the original workspace

The optional script `scripts/export_legacy_data.py` reads the old workspace's frozen image banks, split archives, and identity/hash records, and writes the portable manifest layout:

```bash
python scripts/export_legacy_data.py --rank-root /path/to/Rank --output data
```

The export helper is not a raw-data preparation or feature-extraction pipeline. It preserves the existing historical split and gene order. Optional `--sections Rep1 Rep2` selects sections; `--bank-mode copy` is the default. The optional `--bank-mode hardlink` shares the bank file's underlying storage with the original file, so both copies must remain read-only. Existing files with different content are not silently overwritten.

An optional top-level manifest `checksums` object maps relative input paths to SHA-256 values. The export helper writes these checksums, and the loader verifies supplied checksums before using the files. Normal operation after export uses only the portable data files. Dataset export was not executed during this packaging step.
