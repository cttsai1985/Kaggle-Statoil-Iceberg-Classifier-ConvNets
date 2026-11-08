"""Explicit client integration with an existing MLflow deployment."""

import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

from iceberg import MLFLOW_EXPERIMENT, PROJECT_ID, REPOSITORY
from iceberg.data import sha256
from iceberg.runtime import verify_run, write_json


def source_commit(manifest: dict) -> str:
    """Use a wheel build receipt only when its package fingerprint matches the producer."""
    source = manifest["source"]
    if source.get("git_commit"):
        return source["git_commit"]
    bundle = manifest.get("code_bundle", {})
    if bundle.get("python_source_sha256") == source["python_source_sha256"]:
        return bundle.get("git_commit") or "unavailable"
    return "unavailable"


def track(run_dir: Path, tracking_uri: str) -> dict:
    """Import one completed run; same logical id + manifest hash is idempotent."""
    import mlflow
    from mlflow import MlflowClient

    if urlparse(tracking_uri).username:
        raise ValueError("Use runtime credentials, not credentials embedded in the tracking URI")
    manifest = verify_run(run_dir)
    if manifest.get("project_id", PROJECT_ID) != PROJECT_ID:
        raise ValueError("Run belongs to a different project")
    logical_id = manifest["logical_run_id"]
    uuid.UUID(logical_id)
    digest = sha256(run_dir / "manifest.json")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = MLFLOW_EXPERIMENT
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment {experiment_name} must be provisioned by mlops-platform first"
        )
    mlflow.set_experiment(experiment_id=experiment.experiment_id)
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
        run_name=f"{manifest['profile']}-s{manifest['config']['seed']}-{logical_id[:8]}",
        tags={
            "logical_run_id": logical_id,
            "manifest_sha256": digest,
            "run_kind": manifest["run_kind"],
            "project_id": PROJECT_ID,
            "repository": REPOSITORY,
            "source_commit": source_commit(manifest),
            "source_sha256": manifest["source"]["python_source_sha256"],
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


def track_cv(run_dir: Path, tracking_uri: str) -> dict:
    """Import a verified CV parent and link every retained fold/seed run beneath it."""
    import mlflow
    from mlflow import MlflowClient

    from iceberg.cv import CV_ARTIFACTS, analyze_cv

    if urlparse(tracking_uri).username:
        raise ValueError("Use runtime credentials, not credentials embedded in the tracking URI")
    analysis = analyze_cv(run_dir)
    manifest = json.loads((run_dir / "cv-manifest.json").read_text())
    if manifest["project_id"] != PROJECT_ID:
        raise ValueError("CV belongs to a different project")
    logical_id = manifest["logical_run_id"]
    uuid.UUID(logical_id)
    digest = sha256(run_dir / "cv-manifest.json")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise ValueError(
            f"Experiment {MLFLOW_EXPERIMENT} must be provisioned by mlops-platform first"
        )
    existing = client.search_runs(
        [experiment.experiment_id], f"tags.logical_run_id = '{logical_id}'"
    )
    if existing:
        if len(existing) != 1 or (
            existing[0].data.tags.get("manifest_sha256") != digest
            or existing[0].info.status != "FINISHED"
        ):
            raise ValueError("Conflicting or incomplete CV import for this logical run")
        return {"status": "already_imported", "mlflow_run_id": existing[0].info.run_id}
    protocol = manifest["protocol"]
    parent = client.create_run(
        experiment.experiment_id,
        run_name=f"cv-{protocol['strategy']}-{logical_id[:8]}",
        tags={
            "project_id": PROJECT_ID,
            "repository": REPOSITORY,
            "logical_run_id": logical_id,
            "manifest_sha256": digest,
            "run_kind": manifest["run_kind"],
            "source_commit": source_commit(manifest),
            "source_sha256": manifest["source"]["python_source_sha256"],
        },
    )
    parent_id = parent.info.run_id
    children = []
    try:
        for key in ["strategy", "folds", "split_seed", "model_seeds", "checkpoint_selection"]:
            client.log_param(parent_id, key, str(protocol[key]))
        for key, value in analysis["pooled_oof"].items():
            if value is not None:
                client.log_metric(parent_id, f"oof.{key}", value)
        for key, value in analysis["seed_distribution"]["log_loss"].items():
            if value is not None:
                client.log_metric(parent_id, f"training_seed.log_loss.{key}", value)
        for key, value in analysis["fold_distribution"]["log_loss"].items():
            if value is not None:
                client.log_metric(parent_id, f"fold.log_loss.{key}", value)
        for filename in sorted(CV_ARTIFACTS | {"cv-manifest.json"}):
            client.log_artifact(parent_id, str(run_dir / filename))
        for record in manifest["fold_runs"]:
            imported = track(run_dir / record["directory"], tracking_uri)
            child_id = imported["mlflow_run_id"]
            previous = client.get_run(child_id).data.tags.get("mlflow.parentRunId")
            if previous is not None and previous != parent_id:
                raise ValueError("Fold run is already linked to a different CV parent")
            client.set_tag(child_id, "mlflow.parentRunId", parent_id)
            client.set_tag(child_id, "cv.fold", str(record["fold"]))
            client.set_tag(child_id, "cv.model_seed", str(record["seed"]))
            children.append({**record, "mlflow_run_id": child_id})
        client.set_terminated(parent_id, status="FINISHED")
    except Exception:
        client.set_terminated(parent_id, status="FAILED")
        raise
    result = {"status": "imported", "mlflow_run_id": parent_id, "fold_runs": children}
    write_json(run_dir / "cv-tracking.json", result)
    return result
