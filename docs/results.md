# Validation and benchmark results

## Modern experiments

All completed holdout experiments use the same seed-25 stratified partition: 1,348 training records and 256 held-out records. Each uses the full training partition for 30 natural epochs; checkpoint selection minimizes holdout loss.

| Model | Inference | Holdout log loss |
|---|---|---:|
| Image-only CNN | Ordinary | 0.218330 |
| Image-only CNN | Four-flip average | 0.227273 |
| CNN with angle/missingness features | Ordinary | 0.208920 |
| CNN with angle/missingness features | Four-flip average | **0.201199** |

Flip averaging hurt the image-only model in this experiment; it is retained as a negative result. These are single-split model-development observations. The [grouped, repeated-seed CV protocol](cv.md) provides the broader validation analysis; its aggregate receipt records completion and scope separately.

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
