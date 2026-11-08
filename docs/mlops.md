# Experiment tracking and shared storage

The shared [mlops-platform](https://github.com/cttsai1985/mlops-platform) repository owns service configuration and operations. Quant research and the competition solutions are clients. This repository owns the model, experiment recipe, portable artifacts and tracking adapter.

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
| GitHub repository | `cttsai1985/kaggle-iceberg-solution-mlops` |
| MLflow experiment | `kaggle/iceberg` |
| Reserved model-registry name | `kaggle-iceberg` |
| Prefect flow/deployment convention | `kaggle-iceberg/local` |
| Portable platform artifact namespace | `projects/iceberg` |

The platform's project registry is the naming authority. Run tags include `project_id`, `logical_run_id`, source revision and source fingerprint. Model registration/promotion and Prefect deployment are explicit operations; this client does not perform them during training.

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

Provision the canonical experiment through the platform bootstrap first; the client refuses to create a differently routed experiment implicitly. The adapter verifies artifact/configuration/split checksums before upload. The logical run UUID and manifest fingerprint make repeated imports idempotent; conflicting or incomplete imports are rejected. Training remains usable when tracking is unavailable.

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
