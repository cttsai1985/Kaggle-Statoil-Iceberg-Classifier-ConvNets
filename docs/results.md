# Validation and benchmark results

## Grouped cross-validation

The fixed grouped-stratified protocol evaluated all 1,604 observations across five folds and two training seeds. All ten models retained their final epoch rather than selecting epochs on outer-fold labels. Known-angle groups have no fit/validation overlap in any fold.

| Metric | Result |
|---|---:|
| Known-angle OOF log loss, 1,471 records | **0.220645** |
| All-row paired-seed OOF log loss, 1,604 records | 0.204792 |
| All-row pooled ROC AUC | 0.974749 |
| Individual-seed pooled log losses | 0.207037 / 0.211528 |
| All-row fold loss range, paired-seed ensemble | 0.176110–0.248654 |
| All-row fold loss sample standard deviation | 0.027443 |

The missing-angle stratum contains 133 ships and no icebergs. Its lower loss makes the overall score more favorable; test observations have known angles, so that cohort is reported first. The weakest fold remains in the report. Two seeds and overlapping training partitions do not establish a confidence interval or universal model stability.

The analyzer verifies the complete parent/child artifact chain and recomputes retained OOF metrics. All ten checkpoints were independently reloaded against the raw training payload. Full test inference averages all ten models; all 8,424 IDs retain official sample order. [The CV protocol](cv.md) explains fit boundaries, acquisition-proxy limitations and artifacts.

## Single-holdout model development

All completed holdout experiments use the same seed-25 stratified partition: 1,348 training records and 256 held-out records. Each uses the full training partition for 30 natural epochs; checkpoint selection minimizes holdout loss.

| Model | Inference | Holdout log loss |
|---|---|---:|
| Image-only CNN | Ordinary | 0.218330 |
| Image-only CNN | Four-flip average | 0.227273 |
| CNN with angle/missingness features | Ordinary | 0.208920 |
| CNN with angle/missingness features | Four-flip average | **0.201199** |

Flip averaging hurt the image-only model in this experiment; it is retained as a negative result. These are single-split model-development observations. The grouped, repeated-seed protocol above provides the broader validation analysis.

## Official competition scoring

| Submission | Public log loss | Private log loss |
|---|---:|---:|
| Historical single model, 2017-10-30 | 0.19348 | 0.19311 |
| Historical single model, 2017-10-30 | 0.20507 | 0.20730 |
| Modern metadata CNN with four-flip inference, 2026-11-08 | 0.21463 | 0.19855 |

Scores were retrieved from authenticated Kaggle submission history. The modern submission is **58138050**, using source **897bd39**, all 8,424 official test IDs in sample-submission order, and a fingerprinted output. It does not beat the best historical result. Local holdout loss and leaderboard loss are different evaluations.

The historical trained checkpoints and complete random states are unavailable. Modern code changes are explicit; this comparison measures observed performance rather than replaying identical old weights.

## Engineering verification

The selected complete real-data run trained 630 optimizer steps and 40,320 sampled examples in 30 epochs. It completed in about 121 seconds on the recorded four-thread CPU host. Training drops a short final minibatch; eligible records are reshuffled each epoch.

Its checkpoint reloaded independently, full test predictions preserved all official IDs, and the completed run imported into the shared `kaggle/iceberg` MLflow experiment. Repeated import returned the same run; a checkpoint downloaded through the SeaweedFS-backed artifact proxy matched its original SHA-256.

[The validation receipt](validation.json) contains source/input/artifact identities and the checked runtime scope. Infrastructure operation belongs to [mlops-platform](https://github.com/cttsai1985/mlops-platform). Hosted execution and CV results are reported only once measured.

## Hosted and container execution

The offline Kaggle metadata-model run completed on Python 3.13 / PyTorch 2.11 / scikit-learn 1.6. Its saved checkpoint reloaded locally with identical retained metrics. That run is an independent single-holdout portability check, not the grouped CV ensemble. Code/config/input identities, exact notebook version and package differences are recorded in [the hosted receipt](hosted-validation.json). Docker independently trained a fixture, evaluated a real checkpoint, enforced Lab authentication and retrieved the shared artifact through its client network.
