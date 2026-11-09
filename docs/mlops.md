# Experiment tracking and shared storage

The shared [ML infrastructure](https://github.com/cttsai1985/ml-infrastructure) repository owns service configuration and operations. Quant research and the competition solutions are clients. This repository owns the model, experiment recipe, portable artifacts and tracking adapter.

| Service | Role |
|---|---|
| Docker | Reproducible runtime and isolated service processes |
| Prefect | Workflow orchestration and task/flow state |
| MLflow | Experiment metrics, lineage and artifact access |
| SeaweedFS | S3-compatible artifact storage behind the MLflow service |
| PostgreSQL | Persistent service metadata |

## Project identity

| Identifier | Value |
|---|---|
| Stable project ID | `iceberg` |
| GitHub repository | `cttsai1985/kaggle-statoil-iceberg-solution` |
| MLflow experiment | `kaggle/iceberg` |
| Reserved model-registry name | `kaggle-iceberg` |
| Prefect flow/deployment convention | `kaggle-iceberg/local` |
| Client-owned artifact namespace | `projects/iceberg` |

This repository owns the identities above; they are client configuration, not entries required in a shared infrastructure catalog. Run tags include `project_id`, `logical_run_id`, source revision and source fingerprint. Model registration/promotion and Prefect deployment are explicit operations; this client does not perform them during training.

Existing quant-research storage and historical experiments retain their identities during adoption. The portable platform configuration is not an instruction to migrate existing buckets or volumes.

## Verify and import a completed run

```bash
uv sync --locked --group dev --group lab --group data --group tracking
.venv/bin/iceberg verify --run-dir .local/my-run \
  --data "$ICEBERG_DATA_DIR/train.json"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5500
.venv/bin/iceberg track --run-dir .local/my-run
```

Port 5500 is the audited existing quant-research MLflow endpoint. Configure the URI for your deployment. Supply authentication through runtime environment variables when required; keep credentials outside tracking URLs and Git.

Before importing, an authorized operator creates the client-declared experiment through MLflow with artifact location `mlflow-artifacts:/projects/iceberg`. The client requires that experiment and refuses to create a differently routed experiment implicitly; no platform project registration is needed. The adapter verifies artifact/configuration/split checksums before upload. The logical run UUID and manifest fingerprint make repeated imports idempotent; conflicting imports are rejected and incomplete imports require explicit recovery. Training remains usable when tracking is unavailable.

An isolated SQLite-backed integration test verifies upload, byte-identical retrieval, duplicate import and conflict rejection. Install the `tracking-test` dependency group to run it. The results receipt records any separately verified shared-service import.

## Container access

The base Lab Compose file works independently. To join an existing platform client network:

```bash
export MLOPS_DOCKER_NETWORK=mlops-platform_clients
export MLFLOW_TRACKING_URI=http://mlflow:5000
docker compose -f compose.yaml -f docker/compose.mlops.yaml up -d lab
```

For the existing quant stack, use `MLOPS_DOCKER_NETWORK=kaggle-quant-collections_ui`. This attaches the client to the MLflow-facing network; it does not change the service or connect the notebook to database/storage administration networks.

Inside a container, localhost refers to that container. A host-gateway route alone cannot reach a service published only on the host's loopback address. Use the explicit client network and the correct service hostname.

## Existing SeaweedFS client mount

```bash
export SEAWEED_MOUNT=/absolute/path/to/existing/client-mount
docker compose -f compose.yaml -f docker/compose.seaweed.yaml up -d lab
```

The mount appears read-only at `/shared`. It must be an existing client-side filesystem mount, not the storage server's internal volume. This override creates neither a FUSE mount nor a server. MLflow's artifact proxy remains a separate route; notebooks need no S3 administration credentials.

## Kaggle and orchestration

An offline Kaggle job writes a portable run bundle. Retrieve it, verify its exact code/input/artifact identities, then import it through the same adapter from a runtime that can reach the tracker. Kaggle does not need access to the private services.

The platform includes orchestration infrastructure and its own integration checks. The current model benchmark's execution context is recorded in its receipt; the presence of a Prefect server does not retroactively make a manually launched experiment a Prefect run. W&B mirroring is not implemented.

## Recover an interrupted tracking import

A tracker outage does not require retraining. The importer validates the unchanged local
bundle, uploads it, and downloads every artifact to verify its SHA-256 before marking the
MLflow run `FINISHED`. A normal repeat also verifies the remote bytes. This adds an artifact
read-back cost, including the full model checkpoint.

If an upload fails, `tracking.json` records its remote run ID and exception type;
`tracking-events.jsonl` retains timestamped attempt transitions. Confirm that the previous
importer has stopped, then explicitly recover the same bundle:

```bash
.venv/bin/iceberg track --run-dir .local/my-run --resume
```

Recovery reuses the original MLflow run ID and uploads the same immutable artifact paths;
changed manifests and duplicate remote identities are rejected. `RUNNING` can mean a live
writer or a crashed importer, so it is never resumed implicitly. This is a single-writer
consumer contract, not a distributed lock. A supplied `PREFECT_FLOW_RUN_ID` links the import
attempt to its orchestrator. Training manifests and checkpoints are not rewritten.

The isolated SQLite integration tests inject a partial-upload interruption and corrupt
read-back bytes, check that completion is blocked, then recover with the same run ID.
These tests verify client behavior; they do not establish availability or recovery
objectives for the shared infrastructure.

For cross-validation, use `iceberg track-cv --run-dir .local/my-cv --resume`.
The existing parent and child IDs are retained; already completed folds are verified and
partially uploaded folds are recovered. Parent completion requires all children to finish.
The parent receipt and attempt journal are `cv-tracking.json` and
`cv-tracking-events.jsonl`.
