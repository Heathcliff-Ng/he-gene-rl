# Running and verifying the project

Full standalone training from scratch and test-set evaluation completed on 2026-11-13 using this package and the external Rank datasets. All 192 comparable aggregate and per-seed metrics match the historical experiment exactly. The compact record is [reports/reproduction_20261113.json](../reports/reproduction_20261113.json); [results and interpretation](results.md) distinguish the new run from historical baselines. The initial 2026-11-02 packaging step only imported historical artifacts.

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

### Rebuild directly from the external Rank datasets

The standalone data preparation code can rebuild the historical cell partitions
directly from `../datasets`. It reads raw counts, UNI features, image coordinates,
and cell/gene identities. It imports no original Rank modules and reads no old
outputs or checkpoints. The fixed spatial boundary and identity-hash sampling
recipe selects 65,536 training, 8,192 policy, 1,024 monitor, 1,024 calibration,
and 4,096 test cells per section.

The completed reproduction used the existing `HERL` Conda environment. The
commands below document that workflow, including the corrected external log
directory. The preparation, training, and evaluation outputs already exist on
the original workstation; choose new output paths for a separate run. The
optional unit-test command is shown for checking the environment, but its
console report is not included in the reproduction record. Start in a terminal:

```bash
conda activate HERL
cd /home/p3/PyCharmProjects/Rank/he-gene-rl
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
set -o pipefail

python -c 'import numpy, scipy, h5py, pytest, torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -m pytest -q

python scripts/prepare_rank_data.py \
  --datasets-root ../datasets --source-variant with-spatialex \
  --output data/reproduction_20261113

mkdir -p runs/logs
python -u -m he_gene_rl train \
  --config configs/default.json \
  --data data/reproduction_20261113/manifest.json \
  --output runs/reproduction_20261113 --device cuda \
  2>&1 | tee -a runs/logs/reproduction_20261113.train.log

python -u -m he_gene_rl evaluate-run \
  --run runs/reproduction_20261113 \
  --data data/reproduction_20261113/manifest.json \
  --output results/reproduction_20261113 --device cuda \
  2>&1 | tee -a runs/logs/reproduction_20261113.evaluate.log

cat results/reproduction_20261113/RESULTS.txt
```

Keep shell logs outside the training and evaluation output directories. Both
commands require an empty directory when initializing a new output; `tee`
can create its log before the command checks that directory.

Proceed to each step only after the preceding command succeeds. The default
configuration trains all three stages from scratch for Rep1/Rep2 and seeds
42/43/44. Evaluation covers both cross-section directions and same-section
controls with five method variants. This is the full historical protocol,
not a small smoke run or a fit over every available cell.

`with-spatialex` selects the exact source filenames used by the historical
pipeline (`*_full_with_spatialex_all_pred.h5ad`); the preparation code never
loads their prediction layers. The default `plain` option selects `*_full.h5ad`,
whose relevant fields have not yet been checked for equivalence. Source H5ad
files are opened read-only. Prepared arrays and their provenance stay under
this project. No new environment is needed; preparation requires `h5py`
(also declared in the optional `data`/`test` dependencies).

The historical PPO costs (`FP + 2 FN`, lower is better), averaged over three
seeds, are **41.3672** for Rep1 to Rep2 and **40.7877** for Rep2 to Rep1.
Compare all FP/FN/cost values and the fixed-rule and one-shot controls in
[the results](results.md), not just these two means. The completed standalone
run reproduced all 192 comparable metrics exactly. This verifies the reported
metrics for the documented dataset and protocol; it does not assert equality
of checkpoint bytes or every per-cell output.

### Generic portable-manifest training

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

## Verification record

| Check | Recorded status on 2026-11-13 |
|---|---|
| Direct H5ad preparation with historical split rules | Completed for Rep1 and Rep2 using internal code |
| Full pretraining, adaptation, and PPO | Completed for both sources and seeds 42/43/44 |
| Frozen test-set evaluation | Completed: four directions, five methods, 60 prediction archives |
| Historical metric comparison | All 192 comparable values identical; maximum absolute difference 0 |
| Unit tests | Console report not archived; no pass count claimed here |
| Interrupted-training resume and byte/per-cell parity checks | Not established by this report |

The metric comparison covers FP, FN, cost, and selected-gene count for
`teacher_sequence`, `adapted_sequence`, and `ppo_sequence`: 48 direction-level
means plus 144 per-seed means. One-shot methods were also evaluated in the new
run, but are outside this 192-value comparison. The historical import records
retain their import-time status; the new report records the completed standalone
run separately. There is no need to rerun training merely to update documentation.

Exact numerical results can depend on software versions, numerical settings,
and the feature bank. This record establishes reproduction on the documented
workstation and data, not on every platform or a new patient cohort.

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

For a new checkout without a remote, add the intended GitHub remote afterward.
An existing checkout can use its configured `origin` and does not need to be
initialized again. The repository uses the [MIT license](../LICENSE).
