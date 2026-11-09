# Experiment architecture and operating decisions

## One implementation across four runtimes

The CLI, analysis notebook, Docker image and Kaggle entry point import `iceberg`. Model and preprocessing logic live in the package; notebooks provide analysis and execution parameters. The Kaggle entry installs the same code as a wheel, accompanied by a source/config receipt.

| Component | Responsibility |
|---|---|
| `data.py` | Validate radar records; fit training-only preprocessing; apply paired augmentation |
| `model.py` | Map radar features and optional angle metadata to a logit |
| `runtime.py` | Train, evaluate, predict and verify portable experiment artifacts |
| `cv.py` | Fit fixed fold/seed models, audit OOF predictions and predict with the ensemble |
| `tracking.py` | Import verified model and CV runs into an existing MLflow deployment |
| `cli.py` | Expose those operations without implementing a second pipeline |

A compact package keeps the experimental path inspectable. The [shared platform](https://github.com/cttsai1985/mlops-platform) owns service deployment, access control, credentials and storage administration. Each client owns its project/experiment names and supplies its configuration through generic interfaces. This importer requires its declared MLflow experiment to exist, without requiring a platform project-catalog entry. An external scheduler can invoke the CLI; model execution does not require a scheduler or tracking server.

## Data and evaluation boundaries

The fixed stratified split is constructed before fitting channel statistics or angle imputation. Checkpoints carry the preprocessing state and holdout indices; inference never refits them. Image augmentation acts only during training and shares geometry across both polarizations.

The fixed holdout supports comparison with the historical approach, but repeated acquisition angles can overlap partitions. Each manifest records that overlap. The primary evaluation uses five stratified folds grouped by observed angle and two fixed training seeds. Angle is an acquisition proxy, so this still does not establish performance on independently verified unseen scenes.

The separate development holdout chooses a checkpoint; primary grouped CV retains the final checkpoint after 30 fixed epochs, without using outer-fold labels to select an epoch. Kaggle evaluates a submission on its own scored test observations. Both metrics retain their scope. The documented historical score is a benchmark target, not evidence that a new model has already matched it.

## Portable artifact contract

A successful run contains six fingerprinted artifacts: `model.pt`, `config.json`, `split.csv`, `history.json`, `metrics.json` and `validation.csv`. Its manifest records schema version, logical UUID, code/input/config/split identities, environment, preprocessing, training budget and completion status.

`iceberg verify` and MLflow import use the same verifier. It checks the completed schema, expected artifact set, checksums and configuration/split consistency. Supplying the training file also checks its identity. These hashes establish consistency of the supplied bundle; access control and trusted origin remain responsibilities of the storage and execution environment.

Prediction writes a separate receipt containing input, checkpoint and output checksums plus the inference augmentation setting. This binds a submission to a concrete model and recipe.

## Failure and recovery

Training creates a new run directory and records its status before computation. Exceptions and interruption retain failed/interrupted manifests. A completed directory is immutable experimental evidence; start a new run to change configuration or code. Automatic checkpoint resumption is not implemented.

Tracking is an explicit operation after completion. A repeated logical run with the same manifest returns the existing MLflow run. Conflicting or incomplete imports are rejected for inspection rather than silently duplicated. Tracking availability cannot erase a completed local experiment.

## Runtime and dependency choices

`uv.lock` defines the local/container dependency set. Hosted Kaggle supplies its own scientific environment, so its Python/package/device versions and wheel identity are recorded separately. CPU and CUDA are selected explicitly or through `auto`; configuration records seeds, deterministic mode and mixed precision. Cross-hardware bitwise equality is not promised.

Lab runs as a non-root user, binds its host port to localhost and requires token authentication. Data mounts are read-only; experiment outputs are writable. The optional SeaweedFS override exposes an existing client mount, while MLflow's artifact service is a separate route. Runtime credentials stay outside code, notebooks and Git.

## Verification and release

Tests exercise leakage boundaries, missing metadata, paired augmentation, eval batch independence, checkpoint reload, output identity, package fingerprints and MLflow import. CI runs lint/format checks and the suite on Python 3.12 and 3.13. Real-data results and runtime checks are recorded separately from synthetic tests.

A release identifies the merged source, locked environment and observed experiment evidence. Code wheels and checksum receipts can be distributed without competition records, predictions, credentials or trained weights. Benchmark and hosted-runtime gaps remain visible in the results record.
