# Iceberg and Ship Classification

Distinguish icebergs from ships using two radar image bands. This project revisits Chia-Ta Tsai's 2017 Statoil/C-CORE Kaggle solution with a modern PyTorch training pipeline and evaluates it against the original competition results.

## Problem and data

Each observation contains two **75 × 75** radar bands, HH and HV polarization, and an incidence angle. The target is the probability that the observation contains an iceberg. The official dataset provides **1,604 labeled observations** and **8,424 test observations**. Predictions are evaluated with log loss, so confident mistakes are costly.

See the [competition data description](https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/data) for field definitions and data access.

## Model

A compact convolutional network learns spatial features from both radar bands together. Four progressively downsampled stages feed global average and maximum pooling; a dense head produces an iceberg probability. Native batch normalization, LayerNorm and AdamW replace the historical training implementation.

Channel scaling is fitted on training records only. Spatial augmentation transforms both channels together. An optional metadata branch combines the image features with a training-imputed, standardized incidence angle and a missingness indicator. The image-only and metadata variants share the same validation split.

The original Keras source is preserved in [`legacy/`](legacy/). [Model and training changes](docs/migration.md) describe the differences from the 2017 method.

## Validation and results

The primary validation workflow uses five stratified folds grouped by observed incidence angle across two fixed training seeds. It retains every fold and reports pooled OOF metrics, fold dispersion, seed sensitivity and missing-metadata strata. The [CV guide](docs/cv.md) explains the protocol and analyzer.

| Grouped CV diagnostic | Observed result |
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

## Reproduce an experiment

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
