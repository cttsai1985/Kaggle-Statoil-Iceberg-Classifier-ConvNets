# Two radar bands and an incidence angle: a reproducible Iceberg CNN

A maintained PyTorch implementation of Chia-Ta Tsai's 2017 Kaggle Statoil/C-CORE solution. The model combines two radar polarizations with incidence-angle metadata. An executable notebook explains the method; the original Keras scripts remain the historical reference.

The initial modern run trained on **128 real records**, evaluated on the original **256-record holdout**, reloaded its checkpoint, and predicted **all 8,424 test records**. This bounded run verifies execution. Historical score parity and Keras/PyTorch numerical equivalence remain unverified. See [observed results](docs/results.md).

## Follow the reasoning

Open [the data science walkthrough](notebooks/01-method-and-rerun.ipynb): inputs → validation rationale → architecture → bounded training → checkpoint reload → error inspection → predictions. Important computations have an explanation before the code and an interpretation after the output.

[Migration rationale](docs/migration.md) identifies preserved behavior and changes. The [original README](docs/legacy-readme.md) and untouched `scripts/` retain the historical workflow at commit [d7f9f46](https://github.com/cttsai1985/Kaggle-Statoil-Iceberg-Classifier-ConvNets/tree/d7f9f46145f067a82d3ddf3a9e6ca134bd098083).

## Rerun locally

Install [uv](https://docs.astral.sh/uv/) and Python 3.12:

```bash
uv sync --locked --group dev --group lab --group data
export ICEBERG_DATA_DIR=/absolute/path/outside/git/iceberg
.venv/bin/iceberg download --data-dir "$ICEBERG_DATA_DIR"
.venv/bin/iceberg validate --data-dir "$ICEBERG_DATA_DIR"
.venv/bin/iceberg train --data-dir "$ICEBERG_DATA_DIR" \
  --config configs/bounded.json --run-dir .local/my-run
.venv/bin/iceberg evaluate --checkpoint .local/my-run/model.pt \
  --data "$ICEBERG_DATA_DIR/train.json"
.venv/bin/iceberg predict --checkpoint .local/my-run/model.pt \
  --data "$ICEBERG_DATA_DIR/test.json" --output .local/my-run/submission.csv
```

Download requires Kaggle authentication and accepted rules. Every training run needs a new directory. The locked environment currently targets CPU/Linux/Python 3.12. `configs/reference.json` declares the larger three-epoch/eight-pass recipe; its full execution is not verified in this first change.

## Open Docker JupyterLab

```bash
.venv/bin/python tools/create_lab_token.py
docker compose up --build -d lab
```

Set `ICEBERG_DATA_DIR` first. Open http://127.0.0.1:8890/lab and use the token stored in the ignored `.local/jupyter_token`. Source/notebooks, read-only data and writable outputs are mounted into the container. The [runbook](docs/reproduction.md) covers Docker, the dev container, local `.venv` and the offline smoke.

Shared infrastructure stays outside this repository. [MLOps connections](docs/mlops.md) use an existing MLflow/SeaweedFS deployment; an optional Compose override exposes an existing client-side SeaweedFS mount. No shared servers are deployed.

## Prepare Kaggle execution

The [thin offline entry point](kaggle/rerun.ipynb) uses the same package. [Packaging instructions](docs/kaggle.md) prepare a code wheel, notebook metadata and portable outputs. Remote execution and competition submission are separate, unverified steps and never automatic.

## Check the maintained implementation

```bash
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src tests tools
.venv/bin/pytest -q
```

Tests protect preprocessing, paired transforms, normalization, holdout identity, reload and output ordering. Legacy scripts are outside the modern lint scope. Data, checkpoints, credentials and prediction rows stay outside Git. Code retains the [MIT license](LICENSE); competition data follows its own terms.
