# Experiment runbook

Python 3.12 and `uv.lock` define the tested CPU/Linux dependency set. The local virtual environment and container environment install compatible locked dependencies independently.

## Acquire authorized data

```bash
uv sync --locked --group dev --group lab --group data
export ICEBERG_DATA_DIR=/absolute/path/outside/git/iceberg
.venv/bin/kaggle auth login
.venv/bin/iceberg download --data-dir "$ICEBERG_DATA_DIR"
.venv/bin/iceberg validate --data-dir "$ICEBERG_DATA_DIR"
```

Accept the competition rules in your account. Download obtains the official train/test/sample 7z archives, extracts their payloads and saves archive/payload SHA-256 receipts. Existing files are treated as a local cache; retain its manifest and do not silently replace cached files with edited data under an official-source claim.

Validation checks unique IDs, finite two-band inputs, angle handling, binary targets and training/test ID separation.

## Offline smoke

```bash
.venv/bin/iceberg fixture --data-dir .local/fixture
.venv/bin/iceberg train --data-dir .local/fixture --config configs/smoke.json \
  --run-dir .local/smoke-run --fixture
.venv/bin/iceberg predict --checkpoint .local/smoke-run/model.pt \
  --data .local/fixture/test.json --output .local/smoke-run/submission.csv
```

Synthetic metrics are fixture evidence. They are not competition results.

## Train, reload and predict on real data

```bash
.venv/bin/iceberg train --data-dir "$ICEBERG_DATA_DIR" \
  --config configs/baseline.json --run-dir .local/my-run
.venv/bin/iceberg evaluate --checkpoint .local/my-run/model.pt \
  --data "$ICEBERG_DATA_DIR/train.json"
.venv/bin/iceberg predict --checkpoint .local/my-run/model.pt \
  --data "$ICEBERG_DATA_DIR/test.json" --output .local/my-run/submission.csv
```

Use a new run directory for every training invocation. `configs/baseline.json` declares the complete modern image-only experiment. `configs/metadata.json` uses the same split with an angle branch.

| Artifact | Purpose |
|---|---|
| manifest.json | Run identity, source/input hashes, environment, scope and artifact checksums |
| config.json | Resolved recipe |
| split.csv | Actual train/validation/unused identities |
| history.json | Per-epoch training diagnostic and holdout result |
| model.pt | Best model state, configuration and holdout indices |
| metrics.json / validation.csv | Reconstructable holdout evidence |
| submission.csv / submission.receipt.json | Explicit test predictions and provenance |

Training cost depends on the experiment profile and runtime. Measured timings and environment details are recorded with each run; full test parsing and inference add time.

## JupyterLab in Docker

```bash
.venv/bin/python scripts/create_lab_token.py
docker compose up --build -d lab
docker compose stop lab
```

Open http://127.0.0.1:8890/lab. View the ignored `.local/jupyter_token` locally at login; do not copy it to Git or notebook outputs. Set `ICEBERG_DATA_DIR` before launching.

The runtime mounts source/notebooks at `/workspace`, read-only data at `/data` and writable outputs at `/outputs`. The host virtual environment is not imported. Lab uses four CPU cores, up to 8 GiB RAM and 256 processes. The supplied container user has UID 1000; adjust user IDs and writable directory permissions on other hosts.

Open the same project in the provided dev container for editor development. It uses the Lab Docker target and requires the host data-directory variable. Tracked notebooks should not live only in ignored scratch.

The CPU runtime target can independently evaluate a saved model:
```bash
docker build -f docker/Dockerfile --target runtime -t iceberg-runtime:0.1 .
docker run --rm --network none --cpus 4 --memory 8g \
  -v "$PWD:/workspace:ro" -v "$ICEBERG_DATA_DIR:/data:ro" iceberg-runtime:0.1 \
  evaluate --checkpoint /workspace/.local/my-run/model.pt --data /data/train.json
```

## Reproducibility limits

Recorded CPU behavior does not establish identical results across arbitrary libraries, hardware or frameworks. Local holdout and leaderboard performance are separate measurements. Consult the results record for the verified experiment, inference settings and runtime evidence.
