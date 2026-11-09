"""Recover an interrupted upload without changing a verified local experiment.

Resume is an explicit operator action: the earlier importer must have stopped.
MLflow tags are evidence, not distributed locks or a uniqueness constraint.
"""

import hashlib
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_remote(client, run_id, artifacts):
    """Read through the configured artifact repository and verify every byte hash."""
    with tempfile.TemporaryDirectory(prefix="mlflow-readback-") as directory:
        for name, expected in artifacts.items():
            downloaded = client.download_artifacts(run_id, name, directory)
            if _sha256(downloaded) != expected:
                raise ValueError(f"Remote artifact checksum mismatch: {name}")


def import_verified(
    client,
    *,
    experiment_id,
    run_dir,
    logical_id,
    manifest_digest,
    artifacts,
    tags,
    run_name,
    params,
    metrics,
    resume=False,
    receipt_name="tracking.json",
    finish_children=None,
):
    """Upload, read back, then finish; explicit recovery reuses the same remote ID.

    The caller has already validated the complete local bundle. Child publication,
    when supplied, is part of completion and must also be repeatable by identity.
    """
    matches = client.search_runs([experiment_id], f"tags.logical_run_id = '{logical_id}'")
    if len(matches) > 1:
        raise ValueError("Conflicting duplicate imports for this logical run")
    previous = matches[0] if matches else None
    if previous and previous.data.tags.get("manifest_sha256") != manifest_digest:
        raise ValueError("Conflicting manifest for this logical run")
    if previous and previous.info.status == "FINISHED":
        verify_remote(client, previous.info.run_id, artifacts)
        children = finish_children(previous.info.run_id, False) if finish_children else {}
        result = {
            "status": "already_imported",
            "mlflow_run_id": previous.info.run_id,
            "remote_artifacts_verified": len(artifacts),
            **children,
        }
        _write_receipt(run_dir, receipt_name, result)
        return result
    if previous and not resume:
        raise ValueError(
            "Incomplete import; confirm the previous importer has stopped, then use --resume"
        )
    attempt = str(uuid.uuid4())
    lineage = {
        "logical_run_id": logical_id,
        "manifest_sha256": manifest_digest,
        "import_attempt_id": attempt,
    }
    flow_id = os.environ.get("PREFECT_FLOW_RUN_ID")
    if flow_id:
        lineage["prefect.flow_run_id"] = flow_id
    if previous:
        run_id = previous.info.run_id
    else:
        run_id = client.create_run(
            experiment_id, run_name=run_name, tags={**tags, **lineage}
        ).info.run_id
    receipt = {
        "mlflow_run_id": run_id,
        "logical_run_id": logical_id,
        "manifest_sha256": manifest_digest,
        "import_attempt_id": attempt,
        "resumed": previous is not None,
        **({"prefect_flow_run_id": flow_id} if flow_id else {}),
    }
    try:
        client.update_run(run_id, status="RUNNING")
        for name, value in {**tags, **lineage, "artifact_verification": "pending"}.items():
            client.set_tag(run_id, name, value)
        _write_receipt(run_dir, receipt_name, {**receipt, "status": "uploading"})
        for name, value in params.items():
            client.log_param(run_id, name, value)
        for name, value in metrics.items():
            client.log_metric(run_id, name, value)
        for name in artifacts:
            parent = Path(name).parent
            client.log_artifact(
                run_id,
                str(run_dir / name),
                artifact_path=None if parent == Path(".") else str(parent),
            )
        verify_remote(client, run_id, artifacts)
        children = finish_children(run_id, resume) if finish_children else {}
        client.set_tag(run_id, "artifact_verification", "verified")
        client.set_terminated(run_id, status="FINISHED")
        result = {
            **receipt,
            "status": "resumed" if previous else "imported",
            "remote_artifacts_verified": len(artifacts),
            **children,
        }
        _write_receipt(run_dir, receipt_name, result)
        return result
    except BaseException as error:
        failure = {**receipt, "status": "failed", "error_type": type(error).__name__}
        # Keep the original upload error if the tracker itself is unreachable.
        try:
            client.set_terminated(run_id, status="FAILED")
        except Exception as status_error:
            failure["remote_status_error_type"] = type(status_error).__name__
        _write_receipt(run_dir, receipt_name, failure)
        raise


def _write_receipt(run_dir, name, result):
    path = run_dir / name
    event = {"recorded_at": datetime.now(UTC).isoformat(), **result}
    with path.with_name(path.stem + "-events.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
