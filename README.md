# Statoil: Iceberg and Ship Classification

Remote marine operations need timely information about possible ice hazards. The [Statoil/C-CORE challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) focuses on one step: distinguish ships from icebergs in supplied radar image candidates. This solution uses paired HH/HV channels, a compact PyTorch CNN and grouped validation to produce iceberg probabilities.

The inference study finds a useful improvement: **four-flip averaging lowers known-angle OOF log loss for both model variants**. Adding incidence-angle metadata looks stronger on the aggregate metric, but **95% of that gain comes from missing-angle training records**, a cohort absent from the test set. That difference changes which result should guide the inference recipe.

The potential operating role is to support an analyst's candidate-review queue. [Problem and value](docs/problem-and-value.md) connects model probabilities to review decisions; the retained workflow generates all **8,424 test predictions** and verifies the saved preprocessing and weights used to produce them.

[Model comparison](docs/modeling-study.md) · [Results](docs/results.md) · [Run the solution](#run-it-yourself)

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

The historical public score **0.19348** corresponds to approximately **top 49.8%** of the downloaded public leaderboard; the modern rerun's **0.21463** corresponds to **top 59.5%**. [Percentile method and snapshot](docs/results.md#leaderboard-percentile-context). These are score-equivalent positions, separate from official late-entry ranks.

## Which observations drive the improvement

A lower aggregate loss initially makes the angle branch look useful. Checking where that improvement comes from changes what the result can support: **95% of the all-row gain comes from 133 missing-angle ships**, while test observations have known angles. On known-angle rows, the small benefit reverses sign between training seeds.

The model comparison therefore follows the population expected at inference. The report therefore retains known-angle and all-row results, both seeds and every fold; the metadata recipe stays a frozen reference alongside the simpler comparison. [Evidence and reasoning](docs/modeling-study.md) · [Executed checks](notebooks/02-controlled-ablations.ipynb).

## Problem and data

Each observation contains two **75 × 75** radar bands, HH and HV polarization, and an incidence angle. The target is the probability that the observation contains an iceberg. The official dataset provides **1,604 labeled observations** and **8,424 test observations**. Predictions are evaluated with log loss, so confident mistakes are costly.

See the [competition data description](https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/data) for field definitions and data access.

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

## Analysis notebook

The [data-science notebook](notebooks/01-method-and-rerun.ipynb) examines radar inputs, missing metadata, validation, model training, holdout predictions and test inference. It uses the same package as the command-line workflow.

## Recovering a prediction batch

In the [controlled batch drill](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/batch-operations.md), model verification and evaluation completed before an unavailable dependency blocked prediction and delivery. Repeating the entire job would mix an external failure with already completed scientific computation.

The runner instead preserves stage identities and output hashes. After the dependency is repaired, it reuses the evaluation, produces all **8,424** predictions in the expected ID order and verifies model bytes retrieved from storage. A separately rejected quality gate blocks delivery while retaining the original evaluation. This explains the separate evaluation, readiness and delivery states—and why retries must have an explicit scope.

This solution consumes shared orchestration, tracking and storage services. The evidence is a saved-model batch drill and client recovery tests, rather than a deployed maritime service. [Consumer contract](docs/consumer-contract.md) · [Capability boundaries](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/mlops-assessment.md) · [Original walkthrough](notebooks/01-method-and-rerun.ipynb).

## Model artifacts and delivery

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
uv sync --locked --group dev --group serving --group tracking --group tracking-test
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/pytest -q
```

Data, trained weights, credentials and prediction rows are excluded from Git.

## Use the saved model

The [model packaging and serving guide](docs/serving.md) explains the actual input/output contract, export from saved artifacts and the shared HTTP parity check. A canary release retains the tested source, dependency lock and aggregate verification evidence; private model weights and request rows are not release assets.

## License

Source code is licensed under the [MIT License](LICENSE). Competition data follows its own terms.
