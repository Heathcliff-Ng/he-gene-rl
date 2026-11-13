# Results and reproduction

The original experiments completed on 2026-11-01. On 2026-11-13, this standalone package completed fresh training and test-set evaluation and reproduced the comparable historical metrics exactly. The two runs are recorded separately below.

## Evaluation definition

The endpoint is the final gene set over all 313 genes. A true count of zero is negative, at least two is positive, and one is neutral. Cost is the mean per-cell `FP + 2 FN`, with lower values preferred. Each direction uses 4,096 held-out cells. HE-Gene-RL models and the historical count+BCE baseline are averaged over seeds 42, 43, and 44. SpatialEx is one existing frozen model with source-calibrated decisions; repeating its scores in seed-aligned comparisons does not constitute three independent SpatialEx fits.

## Standalone reproduction: 2026-11-13

Run `reproduction_20261113` used the package's own data preparation, training, and evaluation code. Data preparation read raw counts, UNI features, image coordinates, and identities from the external Rank datasets, using the historical `*_full_with_spatialex_all_pred.h5ad` filenames without reading prediction layers. It rebuilt the historical partitions without loading old output caches or importing original Rank modules. All three training stages ran from scratch for Rep1 and Rep2 with seeds 42, 43, and 44, producing six trained predictor/policy pairs.

Evaluation covered all four source-to-target directions and all five internal methods, producing 60 prediction archives. Each direction contained 4,096 test cells and 313 genes. All requested predictions finished before the evaluation command loaded test counts for scoring.

The three sequence methods shared with the historical report (`teacher_sequence`, `adapted_sequence`, and `ppo_sequence`) matched on **192 numeric metrics, with a maximum absolute difference of 0**. This comparison includes FP, FN, cost, and selected-gene count for all four directions: 48 seed-averaged metrics and 144 per-seed metrics. This establishes equality of those recorded evaluation metrics; it is not a claim that checkpoint files or every prediction array were compared byte for byte.

The new cross-section costs are:

| Method | Rep1 → Rep2 | Rep2 → Rep1 |
|---|---:|---:|
| Pretrained predictor, predicted-count sequence (`teacher_sequence`) | 42.026449 | 40.900716 |
| Adapted predictor, fixed selection rule (`adapted_sequence`) | 41.445231 | 40.919352 |
| Adapted predictor + PPO (`ppo_sequence`) | 41.367188 | 40.787679 |
| Pretrained predictor, one-shot selection (`teacher_one_shot`) | **41.014486** | **40.589437** |
| Adapted predictor, one-shot selection (`adapted_one_shot`) | 41.277262 | 40.709554 |

PPO reduced cost relative to the adapted fixed-rule sequence by 0.1883% and 0.3218%. The corresponding 95% spatial-block intervals for the cost difference were [-0.160761, -0.002035] and [-0.166949, -0.094622]. The increment is small. Both one-shot variants had lower mean cross-section costs than PPO; the pretrained one-shot variant was best in both directions. These results do not establish an advantage of sequential reinforcement learning over one-shot prediction.

The compact record is [reports/reproduction_20261113.json](../reports/reproduction_20261113.json). Full local artifacts are in `runs/reproduction_20261113/` and `results/reproduction_20261113/`; they remain excluded from the source repository. Reproduction instructions are in [docs/reproduction.md](reproduction.md).

SpatialEx and original STRank were not retrained or reevaluated in this run. Their numbers below remain historical references. This run reused the same specimen and test partitions, so it validates reproduction on those data rather than generalization to new patients.

## Historical cross-section results

| Method | Rep1 → Rep2 FP | FN | Cost | Rep2 → Rep1 FP | FN | Cost |
|---|---:|---:|---:|---:|---:|---:|
| SpatialEx, source calibrated | 16.0718 | 16.5947 | 49.2612 | 16.5637 | 16.4014 | 49.3665 |
| Original STRank count+BCE, source recalibrated | 14.0701 | 14.1928 | 42.4557 | 13.4985 | 14.3300 | 42.1585 |
| Pretrained predictor, predicted-count sequence | 16.1790 | 12.9237 | 42.0264 | 15.8014 | 12.5496 | 40.9007 |
| Adapted predictor, fixed selection rule | 11.2634 | 15.0909 | 41.4452 | 9.3821 | 15.7686 | 40.9194 |
| Adapted predictor + PPO | 11.8535 | 14.7568 | 41.3672 | 9.9377 | 15.4250 | 40.7877 |
| Pretrained predictor, one-shot empty-history selection¹ | 11.1128 | 14.9508 | 41.0145 | 9.5424 | 15.5235 | 40.5894 |

¹ This existing one-shot control uses the same pretrained image/context predictor, predicts all genes from an empty history, and selects genes with positive base utility. It comes from the earlier aligned evaluation, not from the later three-variant PPO run. It is distinct from the adapted predictor and must not be described as a same-weight PPO ablation.

The complete PPO system reduced cost relative to SpatialEx by **16.02%** and **17.38%**, and relative to the original count+BCE baseline by **2.56%** and **3.25%**. These are reductions in the specified cost, not percentage-point gains in prediction accuracy. Relative to the original baseline, FP decreased while FN increased.

## Historical PPO increment on a fixed predictor

The matched comparison holds the adapted predictor fixed and changes only the selection/stopping policy:

| Direction | PPO minus fixed-rule cost | 95% spatial-block interval | Relative cost reduction |
|---|---:|---:|---:|
| Rep1 → Rep2 | -0.0780 | [-0.1608, -0.0020] | 0.19% |
| Rep2 → Rep1 | -0.1317 | [-0.1669, -0.0946] | 0.32% |

All six cross-section seed comparisons improved, and all six selected PPO checkpoints were later than iteration zero. The measured increment is small, and most improvement over the original count+BCE baseline was already present in the adapted predictor. The one-shot pretrained control in the preceding table has a lower cost than the final PPO system in both directions. These results do not establish a general advantage of sequential inference over one-shot prediction.

## Historical same-section supplementary results

| Method | Rep1 → Rep1 cost | Rep2 → Rep2 cost |
|---|---:|---:|
| SpatialEx | 47.6877 | 49.4155 |
| Original count+BCE | 39.7693 | 41.8944 |
| Pretrained predictor, predicted-count sequence | 39.6513 | 41.5116 |
| Adapted predictor, fixed selection rule | 38.4323 | 40.2007 |
| Adapted predictor + PPO | 38.3979 | 40.1545 |

The PPO-versus-adapted intervals cross zero for both same-section directions. These are supplementary results: the original count+BCE model had previously used its same-section evaluation cells for validation, so this is not an independent primary comparison for that baseline.

## Scope and provenance

The two sections are from the same specimen, and the historical test data had been examined during earlier work. The results are exploratory cross-section evidence, not independent patient-level external validation. Spatial uncertainty intervals average training seeds per cell first, then resample spatial blocks with 2,000 replicates. They do not quantify variation across patients.

The original count+BCE and the current system differ in neighbor inputs, architecture, and supervision definitions. Their difference measures a complete-system change and cannot be attributed entirely to PPO or to the count head.

The historical aggregate record is [reports/legacy_summary.json](../reports/legacy_summary.json). Original source records were:

- `experiments/predicted_history_rl_20261030/artifacts_full/RESULTS.txt` and `evaluation_summary.json` for the PPO experiment.
- `experiments/predicted_count_inference_20261030/artifacts_full/RESULTS.txt` for the aligned one-shot comparison.

Those paths identify the source experiment within the original workspace; runtime code in this project does not depend on them. Dataset files, per-cell predictions, and checkpoint binaries are not included in the source release by default.
