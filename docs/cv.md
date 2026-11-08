# Cross-validation and stability

## Validation question

A useful validation protocol measures performance beyond a favorable random holdout. For this dataset, target balance, missing metadata and repeated incidence angles all matter.

The labeled data contain 133 missing-angle records, all ships. The official test payload has no missing angles. Overall OOF loss therefore includes an easier stratum that is absent from test inference. The analyzer reports known-angle and missing-angle performance separately.

Exact incidence angles also repeat across observations. They are an acquisition proxy, not verified scene IDs. The original random holdout permits overlap. The modern grouped strategy keeps each known-angle group in one fold; missing-angle records receive individual group IDs because their acquisition is unknown.

## Fixed protocol

- Five shuffled **StratifiedGroupKFold** partitions, split seed **25**.
- Two predeclared model seeds, **1017 and 2027**, on the same partitions.
- The same metadata model, paired augmentation and four-flip inference recipe for every fit.
- **30 fixed epochs**, retaining the final checkpoint. Outer-fold labels do not choose a training epoch.
- Preprocessing fitted separately on each fold's training rows.
- Every record receives exactly one OOF prediction per seed; no fold or seed is discarded.

Plain target-stratified folds remain available as a comparison. The analyzer records class balance and acquisition overlap for the actual split. Folds and their fingerprint are retained as artifacts: a seed alone is not a cross-version guarantee of identical library behavior.

The recipe was informed by earlier experiments on these data, so this is fixed-recipe validation rather than a pristine external model-selection test. The separately scored Kaggle submission has its own provenance and metric.

## Run and inspect

```bash
.venv/bin/iceberg cv --data-dir "$ICEBERG_DATA_DIR" \
  --config configs/metadata.json --run-dir .local/my-cv \
  --strategy grouped --folds 5 --split-seed 25 \
  --seeds 1017 2027 --workers 2 --device cpu
.venv/bin/iceberg analyze --run-dir .local/my-cv \
  --data "$ICEBERG_DATA_DIR/train.json"
.venv/bin/iceberg predict-cv --run-dir .local/my-cv \
  --data "$ICEBERG_DATA_DIR/test.json" --output .local/my-cv/submission.csv
```

Two workers with the supplied four-thread model configuration request eight CPU threads. Reduce workers on a smaller machine. A CUDA worker runs alone; CPU parallelism is an explicit choice.

## What the analyzer verifies

The parent manifest binds the input, protocol, fold assignments and every child manifest/checkpoint. Analysis checks child artifact fingerprints, retained validation IDs and probabilities, complete OOF coverage and the paired-seed ensemble. Supplying the training file checks its identity too.

The report includes:

| Diagnostic | Interpretation |
|---|---|
| Per-fold class counts, sizes and angle overlap | Whether the split matches its declared constraints |
| Pooled OOF log loss, Brier score and ROC AUC | Performance over all eligible held-out predictions |
| Per-fold metrics, mean, standard deviation and range | Descriptive variation across data partitions |
| Per-seed pooled metrics and range | Optimization variation on identical partitions |
| Known/missing-angle and class-specific metrics | Which strata drive the overall result |
| Fixed-bin calibration and retained high-loss errors | Probability behavior and concrete failure cases |

Fold standard deviation is **not a confidence interval**: training partitions overlap. Two training seeds provide a limited sensitivity check. No calibrator is fitted to OOF targets, and diagnostic bins do not establish calibration.

## Artifacts and inference

`cv-manifest.json` references each fold/seed experiment. `folds.csv` stores assignments; `oof.csv` retains both seeds; `ensemble-oof.csv` averages their eligible predictions. `analysis.json` is the aggregate report; `errors.csv` retains errors for inspection. Data-derived rows remain outside Git.

For OOF analysis, a record is averaged only over models that held it out. For new test data, `predict-cv` uses all retained fold/seed models with their own preprocessing and the recorded inference policy. Its receipt binds the input, output, CV manifest, every checkpoint and the consumer code revision.

`iceberg track-cv` imports the verified aggregate and its fold runs into `kaggle/iceberg`, with parent/child links, source tags and artifact checksums. The platform provisions that experiment; training does not need a live tracker.

[Observed results](results.md) · [Architecture](architecture.md) · [Tracking](mlops.md)
