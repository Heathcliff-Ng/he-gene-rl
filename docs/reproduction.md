# Running and verifying the project

This project separates source packaging, importing historical artifacts, training, prediction, and evaluation. Packaging has been performed without executing tests, training, or inference. The commands below are instructions for the later verification stage, not a log of completed checks.

## Install the source package

From the project root, use an environment compatible with [pyproject.toml](../pyproject.toml). Install a PyTorch build appropriate for the available CPU/CUDA environment, then:

```bash
python -m pip install -e '.[test]'
```

The installed command is `he-gene-rl`. The Python package is `he_gene_rl`; it does not import `rank` or `experiments` from the original workspace.

## Choose a data and model source

There are two workflows:

| Workflow | Data | Models | Purpose |
|---|---|---|---|
| Historical migration | Export the existing frozen splits and banks | Import existing checkpoints | Later compare the reorganized inference with saved historical predictions |
| New training | Prepare the portable data manifest | Train all three stages | Run the method on a supplied dataset |

If the original Rank directory is the parent of this project, the migration commands are:

```bash
python scripts/export_legacy_data.py --rank-root .. --output data
python scripts/import_legacy_artifacts.py --rank-root .. --output runs/legacy
```

Migration is local file preparation, not a new training run or a new evaluation. A new installation obtained from GitHub will not have these original artifacts; supply a dataset and train, or obtain separately distributed compatible checkpoints.

Imported historical runs are read-only inference/evaluation bundles. The importer copies the selected best checkpoints, source normalization, split identities, and training summaries; it does not provide a new optimizer-resume state or authorize modifying the historical run. The per-source layout is:

```text
runs/legacy/
  run.json
  training_frozen.json
  training_summary.json
  Rep1/
    genes.json
    normalization.npz
    split_ids.npz
    seed42/
      pretraining/best.pt
      adaptation/best.pt
      ppo/best.pt
```

The same layout is repeated for each configured source and seed. Stage summaries live next to their checkpoints. `adaptation/best.pt` and `ppo/best.pt` form the final deployed pair; `pretraining/best.pt` supplies the pretrained comparison variants.

## Train a new run

```bash
he-gene-rl train \
  --config configs/default.json \
  --data data/manifest.json \
  --output runs/main
```

To run individual stages in the same output directory:

```bash
he-gene-rl train --config configs/default.json --data data/manifest.json --output runs/main --stage pretrain
he-gene-rl train --config configs/default.json --data data/manifest.json --output runs/main --stage adapt
he-gene-rl train --config configs/default.json --data data/manifest.json --output runs/main --stage ppo
```

Read the source/seed configuration before starting. The historical settings use both Rep1 and Rep2 as sources with seeds 42, 43, and 44. Each source has its own normalization and separately fitted models. Training is performed on source data; the target section is chosen only at prediction/evaluation time.

The predictor stages and PPO support checkpointed progress. Retain the configuration, split identities, and checkpoint records alongside a run. Use a new output directory when changing a method or configuration rather than mixing incompatible histories.

## Predict without expression

```bash
he-gene-rl predict \
  --run runs/main --data data/manifest.json \
  --source Rep1 --target Rep2 --seed 42 --split test \
  --output predictions/Rep1_to_Rep2_seed42.npz --device cpu
```

The run supplies the predictor, policy, and source normalization. The target split supplies image identities and bank rows. Use `--split predict` for a separately named unlabeled split. No count labels are needed or loaded by prediction.

The saved predictions include the final selection mask, gene-selection sequence, selected-gene predicted values, stopping steps, and aligned cell/gene identities. Selected-gene predicted values are the stored values from each gene's selection time, not a newly measured expression vector or necessarily the predictor's final output for all genes.

The default method is `ppo_sequence`. `--method adapted_sequence` uses the same adapted predictor with its fixed utility selection rule; `teacher_sequence` uses the pretrained predictor with that rule. `teacher_one_shot` and `adapted_one_shot` select all positive-utility genes from empty-history predictions. One-shot outputs contain a full-panel `predicted_counts` array rather than a selection sequence. `--batch-size` defaults to 512 and can be changed independently of the frozen weights.

## Score existing predictions

```bash
he-gene-rl evaluate \
  --predictions predictions/Rep1_to_Rep2_seed42.npz \
  --data data/manifest.json --target Rep2 --split test \
  --output results/Rep1_to_Rep2_seed42.json
```

This is the step that reads true expression. Evaluate the complete final set over the whole panel, including errors made in earlier committed selections. Report FP and FN separately as well as their weighted sum.

The run-level entry point performs the configured frozen evaluations:

```bash
he-gene-rl evaluate-run \
  --run runs/main --data data/manifest.json \
  --output results/main --device cuda
```

The default includes all five method variants and all configured sources, targets, and seeds. A subset can be requested with, for example, `--methods ppo_sequence adapted_sequence`. Every requested prediction finishes before target counts are read for scoring. The result directory contains `predictions/`, `evaluation_summary.json`, and `RESULTS.txt`. Use a new or empty result directory; completed evaluations are not silently overwritten.

## Compare an external baseline

Save the external method's final decisions in the [documented archive format](data_format.md#prediction-and-external-baseline-archives), then run:

```bash
he-gene-rl compare \
  --candidate predictions/Rep1_to_Rep2_seed42.npz \
  --reference predictions/external_baseline.npz \
  --data data/manifest.json --target Rep2 --split test \
  --output results/ppo_vs_external.json \
  --bootstrap-replicates 2000
```

The required fields are a boolean `selected_mask` or `selected`, unique `cell_ids`, and the complete ordered `gene_ids`. Both archives must cover the same evaluation cell set; cell ordering is aligned automatically. Gene ordering must already match. Choose external baseline thresholds or other decision settings on source data before evaluating the target split.

Differences are candidate minus reference, so a negative cost difference favors the candidate. Spatial-block intervals require the dataset's `tiles` and more than one block. With unavailable blocks, point estimates are still reported without confidence intervals. A `compare` call compares one candidate archive with one reference archive; it does not automatically merge multiple training seeds. Run-level evaluation averages matched internal-method seeds per cell before block resampling.

Both `evaluate` and `compare` accept `--neighbors`, defaulting to eight, for the data loader. Their scores use completed selections, not newly generated predictions. `compare` additionally accepts `--bootstrap-replicates`, defaulting to 2,000.

SpatialEx and original STRank historical baseline scores are documented but their training implementations are not bundled into this standalone method package. A new dataset requires new, appropriately aligned baseline outputs before making a new comparison claim.

## Planned verification order

No items in this list are marked complete for the reorganized project:

1. Run `python -m pytest` to check the migrated unit tests and portable data contracts.
2. Use a small source-only dataset to exercise pretraining, adaptation, checkpoint resume, and PPO.
3. Compare imported historical models on a fixed small image batch against the saved original implementation, including gene identities, selected sets, sequence order, and predicted history values.
4. Evaluate complete imported historical runs only after establishing the source and input correspondence.
5. Train a new formal run, with all model selection completed before examining its test results.

The old experiment's exact numerical results can depend on software versions, numerical settings, and the original feature bank. Record the environment and verify parity before claiming exact reproduction. The old adaptation-plus-PPO run took approximately 10.5 minutes on its original hardware; that excludes the prior predictor pretraining and is not a runtime guarantee for a new run.

## Preparing a source release

Keep source, configuration, tests, documentation, and the small aggregate report in version control. Local image banks, expression targets, checkpoint binaries, and per-cell prediction archives belong in the ignored data/run/output locations. The package does not create or push a GitHub repository.

Create a source-only archive with:

```bash
python scripts/build_source_archive.py --output dist/he-gene-rl-source.zip
```

The archive contains only allowlisted source, configuration, tests, documentation,
and aggregate reports. It excludes `runs/legacy`, data, and per-cell predictions.
The included GitHub Actions workflow has a manual `workflow_dispatch` trigger
only, so uploading the repository will not start testing automatically.

For a new repository, extract the source archive into its own directory and
initialize Git there. This avoids adding unrelated files from a parent checkout:

```bash
cd he-gene-rl
git init
git add .
git commit -m "Initial HE-Gene-RL project"
```

Add your own GitHub remote afterward. No remote URL is configured by these files.

Choose a repository license and supply any desired authorship/citation information explicitly. No publication identifier, license grant, or external download URL has been invented during this reorganization.
