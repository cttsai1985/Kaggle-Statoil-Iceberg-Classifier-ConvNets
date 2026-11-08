"""Explicit client integration with an existing MLflow deployment."""

import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

from iceberg.data import sha256
from iceberg.runtime import write_json


def track(run_dir: Path, tracking_uri: str) -> dict:
    """Import one completed run; same logical id + manifest hash is idempotent."""
    import mlflow
    from mlflow import MlflowClient

    if urlparse(tracking_uri).username:
        raise ValueError("Use runtime credentials, not credentials embedded in the tracking URI")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Only completed runs can be imported")
    allowed = {
        "model.pt",
        "config.json",
        "split.csv",
        "history.json",
        "metrics.json",
        "validation.csv",
    }
    if set(manifest["artifacts"]) != allowed:
        raise ValueError("Unexpected artifact manifest")
    for filename, expected in manifest["artifacts"].items():
        if (run_dir / filename).is_symlink():
            raise ValueError("Artifact symlinks are not accepted")
        if sha256(run_dir / filename) != expected:
            raise ValueError(f"Artifact checksum mismatch: {filename}")
    logical_id = manifest["logical_run_id"]
    uuid.UUID(logical_id)
    digest = sha256(run_dir / "manifest.json")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "kaggle/statoil-iceberg-classifier-challenge"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    existing = client.search_runs(
        [experiment.experiment_id], f"tags.logical_run_id = '{logical_id}'"
    )
    if existing:
        previous = existing[0]
        if (
            previous.data.tags.get("manifest_sha256") != digest
            or previous.info.status != "FINISHED"
        ):
            raise ValueError("Conflicting or incomplete import for this logical run")
        return {"status": "already_imported", "mlflow_run_id": previous.info.run_id}
    with mlflow.start_run(
        run_name=manifest["profile"],
        tags={
            "logical_run_id": logical_id,
            "manifest_sha256": digest,
            "run_kind": manifest["run_kind"],
            "legacy_commit": manifest["source"]["legacy_commit"],
        },
    ) as active:
        mlflow.log_params(manifest["config"])
        mlflow.log_metrics(json.loads((run_dir / "metrics.json").read_text()))
        # Artifact proxy routes to the configured shared object store; no server is deployed here.
        for filename in [*manifest["artifacts"], "manifest.json"]:
            mlflow.log_artifact(str(run_dir / filename))
        result = {"status": "imported", "mlflow_run_id": active.info.run_id}
    write_json(run_dir / "tracking.json", result)
    return result
