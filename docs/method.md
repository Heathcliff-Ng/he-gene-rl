# Method and training protocol

This document describes the implemented predictor and PPO policy. The standalone package is a reorganization of the historical experiment; package-level verification is pending.

## Inputs and state

The historical configuration uses 1,024-dimensional cached UNI embeddings, eight spatial image neighbors, 313 genes, and a hidden width of 128. The input feature dimension and gene identities must agree between the dataset and its trained model.

At step `t`, the predictor receives:

- The center cell's image embedding.
- Neighbor image embeddings and their positions relative to the center.
- A vector of selected-gene identities, a vector of available values, and the corresponding count values.

The history is represented by fixed-length gene-indexed vectors, not an RNA measurement stream acquired during deployment. Selected values undergo `log1p` and source-training normalization. Selection and value-availability masks distinguish an unavailable value from a measured or predicted zero.

The image projection maps center and neighbor embeddings to the hidden space. Each neighbor token combines its image projection, relative-position embedding, and image-feature contrast with the center. Center and history representations form an attention query over the center and neighbor tokens. The predictor fuses the center, history, attended context, and a summary of neighborhood feature dispersion.

Neighbor RNA is never an input. In the historical experiment, the image bank covered the entire section, so image neighbors could cross training/test split boundaries. This is access to held-out image context, not held-out expression; use the same convention when reproducing those results.

## Predictor outputs and loss

One network has two heads. For every gene `g`, the state head outputs three probabilities:

\[
p_{g,0}=P(c_g=0),\quad p_{g,1}=P(c_g=1),\quad p_{g,2}=P(c_g\ge2).
\]

Its base selection utility is

\[
u_g=2p_{g,2}-p_{g,0}.
\]

The regression head produces a nonnegative log-count value through softplus. A selected gene receives the continuous point estimate

\[
\hat c_g=\exp(\hat\ell_g)-1.
\]

The count estimate is not rounded. Because the head is fitted in log space, this transformed point estimate should not be described as the unconditional mean of the raw-count distribution. The heads provide per-gene predictions; they are not a joint generative count model.

For the current set of remaining target genes, the supervised objective is

\[
L_{\mathrm{pred}}=L_{\mathrm{3class\ CE}}+0.1L_{\mathrm{logcount\ MSE}}.
\]

Count-one targets have their own neutral CE class. They are excluded from the count MSE and from binary detection metrics. The recorded detection BCE is a diagnostic, not an additional optimized loss term. For each training batch, empty-history and conditional-history losses are evaluated on the same remaining-gene scope and averaged with equal weight.

The cached UNI backbone is not updated by this project. Its downstream image projection, attention, history encoder, fusion layers, and both prediction heads are trained together during the predictor stages.

## Stage 1: conditional pretraining

Randomly choose a legal set of history genes and a history length from `0, 4, 8, 16, 32, 64`. History identities are sampled without consulting their count outcomes. Usually the selected genes receive their measured source-training counts; 15% of samples receive identities without values. Supervise the unselected genes using both empty and conditional inputs.

This is supervised training on randomly constructed states. It is not a sequential RL rollout with measurements revealed after actions. The historical source monitor criterion averages three-class CE across empty and measured-history conditions at fixed history budgets.

The historical settings use AdamW, batch size 256, learning rate `3e-4`, weight decay `1e-4`, and a maximum of 60 epochs with early stopping. Source-training statistics initialize the count normalization and head biases. No target-section statistics are fitted.

## Stage 2: predicted-history adaptation

Initialize from the selected pretrained predictor. For each batch, sample requested prefix lengths from the same history-length list. Generate prefixes using the current predictor without gradients:

1. Predict state probabilities and count point values.
2. Select the highest-utility unselected gene if its utility is positive; otherwise stop.
3. Store that gene's current predicted count and update its selected mask.
4. Continue until stopping or reaching the requested prefix length.

No true counts are consulted during prefix generation. Only afterward are source-training counts read to compute supervised losses for the remaining genes. Update the entire predictor with the equally weighted empty-history and predicted-history losses. This is adaptation from pretrained weights, not teacher/student distillation.

The historical settings use batch size 512, learning rate `1e-4`, a maximum of 12 epochs, and source-monitor early stopping. Select the predictor by the complete predicted-history sequence's `FP + 2 FN`, allowing the initial predictor to remain best.

## Stage 3: PPO

Freeze the adapted predictor. Train only the actor/critic using a separate source policy split. Policy features include the center encoding, current state probabilities and log-count estimates, normalized selected count estimates, selected mask, and selected fraction.

The policy learns residual gene utilities and a stopping component. It uses a factored distribution: a stopping Bernoulli followed, on continuation, by a categorical choice over unselected genes. Its initialization reproduces the adapted predictor's fixed utility rule. Training samples actions; deterministic evaluation first applies the stopping decision and otherwise chooses the highest-ranked remaining gene. Exhausting the panel forces STOP.

Build each complete trajectory using image inputs and predicted count updates only. The reward adapter then reads source-training expression and assigns:

| Action/outcome | Reward |
|---|---:|
| SELECT with true count `>= 2` | +2 |
| SELECT with true count `0` | -1 |
| SELECT with true count `1` | 0 |
| STOP | 0 |

With `gamma = 1`,

\[
R=2TP-FP=2N_{\mathrm{positive}}-(FP+2FN).
\]

The number of true positive genes is constant for a given cell. There is no additional terminal FN penalty, measurement cost, information-gain term, or predictor-loss reward. Rewards do not enter subsequent policy states.

The historical PPO configuration uses clipping `0.2`, GAE lambda `0.95`, four optimization epochs per iteration, learning rate `1e-4`, entropy weight `0.002`, and a maximum of 30 iterations with early stopping. The source monitor criterion is complete deterministic-sequence `FP + 2 FN`. Iteration zero is eligible; selecting it would mean that the monitor did not identify an improvement from PPO updates.

## Deployment and evaluation

Load one adapted predictor, its corresponding PPO policy, and the source normalization. At deployment, both networks are frozen. Begin with an empty history, commit each selected gene, store its pre-action predicted count, and stop permanently when the policy chooses STOP. No hidden expression or per-step true reward is available to inference.

Score the final set separately:

- FP: selected genes whose true count is zero.
- FN: unselected genes whose true count is at least two.
- Cost: `FP + 2 FN` over the full gene panel.
- Count-one genes: neutral for FP/FN and reward; they still count as selections if chosen.

Historical runs used 65,536 predictor-training cells, 8,192 policy-training cells, and 1,024 monitor cells per source. Each test direction contained 4,096 cells. The two source sections and seeds 42/43/44 produced six trained model pairs. The predictor is shared across all genes; there is no separate network per gene.

The system comparison, the effect of predictor adaptation, and the incremental effect of PPO answer different questions. Report the final system's performance and the matched predictor/PPO comparison separately. Historical results and their scope are in [results.md](results.md).
