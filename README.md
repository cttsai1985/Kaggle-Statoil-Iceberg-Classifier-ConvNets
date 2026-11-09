# Iceberg and Ship Classification

A modern revisit of a 2017 Kaggle radar classifier: **the main result is an explicit test of what the model learns, how much results vary, and how to reproduce that evidence.** Two radar bands distinguish icebergs from ships; the engineering work makes each prediction traceable to its data, code, preprocessing and saved model.

| What matters | What the evidence says |
|---|---|
| Competition context | The modern public score **0.21463** corresponds to approximately **top 59.5%** of the downloaded public leaderboard; the historical 0.19348 corresponds to **top 49.8%**. These are score-equivalent positions, not new official competition ranks. |
| Controlled ablation | Adding angle metadata changes known-angle ensemble loss by only **−0.00084**, with opposite effects across the two seeds. Most all-row improvement comes from missing-angle ships, which are absent from test. |
| More consistent component | Four-flip inference improves known-angle loss for both retained seeds in each architecture, though not every fold. |
| Main limitation | All 133 missing-angle training records are ships, while test records have known angles. Known-angle OOF loss (**0.2206**) is more relevant to the test cohort than the more flattering all-row loss (**0.2048**). |

The modern model does not beat the strongest historical submission. The contribution is a clear account of the modeling hypothesis, component contributions, variability and remaining uncertainty, followed by an experiment another person can inspect and operate. [Leaderboard percentile method and snapshot](docs/results.md#leaderboard-percentile-context) · [Executed walkthrough](notebooks/01-method-and-rerun.ipynb).

The [matched ablation study](docs/modeling-study.md) separates component contribution, training-seed sensitivity and cohort effects. It does **not** establish that the metadata branch is superior. [Executed ablation notebook](notebooks/02-controlled-ablations.ipynb).

## From applied science to an operated experiment

[Consumer contract](docs/consumer-contract.md) · [Controlled modeling study](docs/modeling-study.md)

The package supports the same training and inference path locally, in Docker and in an offline Kaggle notebook. Saved runs retain their split, fitted preprocessing and model fingerprints. The [shared operations case study](https://github.com/cttsai1985/mlops-platform/blob/main/docs/batch-operations.md) covers workflow execution, logs, quality gates and recovery; [the capability assessment](https://github.com/cttsai1985/mlops-platform/blob/main/docs/mlops-assessment.md) separates demonstrated controls from remaining production requirements.

This solution is a **consumer of MLOps capabilities** for batch research. Containerization and MLflow tracking are supporting components; neither alone establishes production MLOps. There is no live maritime service, continuously arriving sensor feed or automatic production retraining.

## Problem and data

Each observation contains two **75 × 75** radar bands, HH and HV polarization, and an incidence angle. The target is the probability that the observation contains an iceberg. The official dataset provides **1,604 labeled observations** and **8,424 test observations**. Predictions are evaluated with log loss, so confident mistakes are costly.

See the [competition data description](https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/data) for field definitions and data access.

## Model

A compact convolutional network learns spatial features from both radar bands together. Four progressively downsampled stages feed global average and maximum pooling; a dense head produces an iceberg probability. Native batch normalization, LayerNorm and AdamW replace the historical training implementation.

Channel scaling is fitted on training records only. Spatial augmentation transforms both channels together. An optional metadata branch combines the image features with a training-imputed, standardized incidence angle and a missingness indicator. The image-only and metadata variants share the same validation split.

The original Keras source is preserved in [`legacy/`](legacy/). [Model and training changes](docs/migration.md) describe the differences from the 2017 method.

## Validation and results

The primary validation workflow uses five stratified folds grouped by observed incidence angle across two fixed training seeds. It retains every fold and reports pooled OOF metrics, fold dispersion, seed sensitivity and missing-metadata strata. The [CV guide](docs/cv.md) explains the protocol and analyzer.

| Recorded metadata-reference CV diagnostic | Observed result |
|---|---:|
| Known-angle OOF log loss (test metadata availability) | **0.2206** |
| All-row paired-seed OOF log loss | 0.2048 |
| All-row training-seed pooled-loss range | 0.0045 |
| Known-angle overlap between fit/validation | 0 in every fold |
| OOF coverage | All 1,604 records, once per seed |

All 133 missing-angle records are ships, while test records have known angles. The analyzer keeps both strata visible; known-angle performance is the more relevant test-cohort diagnostic. Folds and seeds are retained, including the weakest fold. Reported dispersion is descriptive, not a confidence interval.


The original single-model submissions scored **0.19348** and **0.20507** on the public leaderboard. These provide the historical benchmark. Current experiment results and their evaluation scope are recorded in [benchmark results](docs/results.md), with fingerprints in the [validation receipt](docs/validation.json).

A separate model-development experiment uses the original **256-record holdout**. Its metadata CNN scored **0.2012 holdout log loss**, followed by **0.21463 public / 0.19855 private** on Kaggle. It does not beat the best historical result; these scores are reported separately from the grouped CV.

## Analysis notebook

The [data-science notebook](notebooks/01-method-and-rerun.ipynb) examines radar inputs, missing metadata, validation, model training, holdout predictions and test inference. It uses the same package as the command-line workflow.

## Run it yourself

The command below reproduces the **frozen metadata reference**, rather than selecting a winner from the ablation. `configs/image-only-flips.json` provides the equally budgeted image-only comparison; use a separate run directory.

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --locked --group dev --group lab --group data
export ICEBERG_DATA_DIR=/absolute/path/outside/git/iceberg
.venv/bin/kaggle auth login
.venv/bin/iceberg download --data-dir "$ICEBERG_DATA_DIR"
.venv/bin/iceberg validate --data-dir "$ICEBERG_DATA_DIR"
.venv/bin/iceberg cv --data-dir "$ICEBERG_DATA_DIR" \
  --config configs/metadata.json --run-dir .local/my-cv \
  --strategy grouped --folds 5 --split-seed 25 \
  --seeds 1017 2027 --workers 2 --device cpu
.venv/bin/iceberg analyze --run-dir .local/my-cv \
  --data "$ICEBERG_DATA_DIR/train.json"
.venv/bin/iceberg predict-cv --run-dir .local/my-cv \
  --data "$ICEBERG_DATA_DIR/test.json" --output .local/my-cv/submission.csv
```

Download requires Kaggle authentication and accepted competition rules. Each training invocation needs a new run directory. The [runbook](docs/reproduction.md) covers profiles, runtime requirements and saved artifacts.

### Docker JupyterLab and dev container

With `ICEBERG_DATA_DIR` set:

```bash
.venv/bin/python scripts/create_lab_token.py
docker compose up --build -d lab
```

Open http://127.0.0.1:8890/lab using the token in the ignored `.local/jupyter_token` file. The container mounts project files, read-only data and writable experiment outputs. The dev container uses the same Docker image.

### Kaggle notebook

The [Kaggle entry notebook](notebooks/kaggle/rerun.ipynb) runs the package from an attached code wheel and saves predictions and experiment artifacts. See [Kaggle execution](docs/kaggle.md) for packaging, execution and scoring instructions.

## Engineering and experiment lifecycle

The [architecture guide](docs/architecture.md) explains package boundaries, training-only preprocessing, portable artifacts, failure recovery and the release process. Local notebooks, CLI commands, Docker and Kaggle use the same implementation.

### Project layout

```text
src/iceberg/     Data preparation, model, training, evaluation and CLI
configs/         Experiment recipes
notebooks/       Analysis and Kaggle entry notebooks
docs/            Results, methodology and operating guides
tests/           Data, model and workflow checks
docker/          Runtime image and shared-storage mount configuration
scripts/         Small packaging and runtime helpers
legacy/          Preserved 2017 implementation
```

Each experiment saves its configuration, split, source and input identities, metrics and checkpoint checksums. An optional MLflow import connects completed runs to an existing tracker. Shared storage and service deployment remain external; [tracking and storage](docs/mlops.md) explains the integration.

## Development

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/pytest -q
```

Data, trained weights, credentials and prediction rows are excluded from Git.

## License

Source code is licensed under the [MIT License](LICENSE). Competition data follows its own terms.
