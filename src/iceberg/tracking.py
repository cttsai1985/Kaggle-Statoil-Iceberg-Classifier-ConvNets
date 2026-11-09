"""Explicit client integration with an existing MLflow deployment."""

import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

from iceberg import MLFLOW_EXPERIMENT, PROJECT_ID, REPOSITORY
from iceberg.data import sha256
from iceberg.import_support import import_verified
from iceberg.runtime import verify_run


def source_commit(manifest: dict) -> str:
    """Use a wheel build receipt only when its package fingerprint matches the producer."""
    source = manifest["source"]
    if source.get("git_commit"):
        return source["git_commit"]
    bundle = manifest.get("code_bundle", {})
    if bundle.get("python_source_sha256") == source["python_source_sha256"]:
        return bundle.get("git_commit") or "unavailable"
    return "unavailable"


def track(run_dir: Path, tracking_uri: str, *, resume: bool = False) -> dict:
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
            f"Experiment {experiment_name} must be provisioned by an authorized MLflow operator "
            "using the client-owned namespace before import"
        )
    return import_verified(
        client,
        experiment_id=experiment.experiment_id,
        run_dir=run_dir,
        logical_id=logical_id,
        manifest_digest=digest,
        artifacts={**manifest["artifacts"], "manifest.json": digest},
        run_name=f"{manifest['profile']}-s{manifest['config']['seed']}-{logical_id[:8]}",
        tags={
            "run_kind": manifest["run_kind"],
            "project_id": PROJECT_ID,
            "repository": REPOSITORY,
            "source_commit": source_commit(manifest),
            "source_sha256": manifest["source"]["python_source_sha256"],
            "legacy_commit": manifest["source"]["legacy_commit"],
        },
        params=manifest["config"],
        metrics=json.loads((run_dir / "metrics.json").read_text()),
        resume=resume,
    )


def track_cv(run_dir: Path, tracking_uri: str, *, resume: bool = False) -> dict:
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
            f"Experiment {MLFLOW_EXPERIMENT} must be provisioned by an authorized MLflow operator "
            "using the client-owned namespace before import"
        )
    protocol = manifest["protocol"]

    def finish_children(parent_id, resume_children):
        children = []
        for record in manifest["fold_runs"]:
            imported = track(run_dir / record["directory"], tracking_uri, resume=resume_children)
            child_id = imported["mlflow_run_id"]
            previous = client.get_run(child_id).data.tags.get("mlflow.parentRunId")
            if previous is not None and previous != parent_id:
                raise ValueError("Fold run is already linked to a different CV parent")
            client.set_tag(child_id, "mlflow.parentRunId", parent_id)
            client.set_tag(child_id, "cv.fold", str(record["fold"]))
            client.set_tag(child_id, "cv.model_seed", str(record["seed"]))
            children.append({**record, "mlflow_run_id": child_id})
        return {"fold_runs": children}

    metrics = {}
    for prefix, values in (
        ("oof", analysis["pooled_oof"]),
        ("training_seed.log_loss", analysis["seed_distribution"]["log_loss"]),
        ("fold.log_loss", analysis["fold_distribution"]["log_loss"]),
    ):
        metrics.update(
            {f"{prefix}.{key}": value for key, value in values.items() if value is not None}
        )
    return import_verified(
        client,
        experiment_id=experiment.experiment_id,
        run_dir=run_dir,
        logical_id=logical_id,
        manifest_digest=digest,
        artifacts={
            name: sha256(run_dir / name) for name in sorted(CV_ARTIFACTS | {"cv-manifest.json"})
        },
        run_name=f"cv-{protocol['strategy']}-{logical_id[:8]}",
        tags={
            "project_id": PROJECT_ID,
            "repository": REPOSITORY,
            "run_kind": manifest["run_kind"],
            "source_commit": source_commit(manifest),
            "source_sha256": manifest["source"]["python_source_sha256"],
        },
        params={
            key: str(protocol[key])
            for key in ("strategy", "folds", "split_seed", "model_seeds", "checkpoint_selection")
        },
        metrics=metrics,
        resume=resume,
        receipt_name="cv-tracking.json",
        finish_children=finish_children,
    )
