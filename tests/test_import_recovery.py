import hashlib
import json
import uuid

import pytest

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")

from iceberg.import_support import import_verified  # noqa: E402


def test_readback_corruption_blocks_completion_then_explicit_resume_repairs(tmp_path, monkeypatch):
    """An accepted upload is not successful until downloaded bytes match."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = run_dir / "model.bin"
    model.write_bytes(b"immutable fixture model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(
        "recovery-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    logical_id = str(uuid.uuid4())
    kwargs = dict(
        experiment_id=experiment,
        run_dir=run_dir,
        logical_id=logical_id,
        manifest_digest="unchanged-manifest",
        artifacts={"model.bin": digest},
        tags={"run_kind": "infrastructure_fixture"},
        run_name="test",
        params={"fixed_parameter": 1},
        metrics={"fixture_metric": 0.5},
    )
    corrupted = tmp_path / "corrupted.bin"
    corrupted.write_bytes(b"corrupted remote bytes")
    with monkeypatch.context() as fault:
        fault.setattr(client, "download_artifacts", lambda *_args: str(corrupted))
        with pytest.raises(ValueError, match="Remote artifact checksum"):
            import_verified(client, **kwargs)
    failed = client.search_runs([experiment])[0]
    assert failed.info.status == "FAILED"
    assert failed.data.tags["artifact_verification"] == "pending"
    with pytest.raises(ValueError, match="Conflicting"):
        import_verified(client, **{**kwargs, "manifest_digest": "changed"}, resume=True)
    monkeypatch.setenv("PREFECT_FLOW_RUN_ID", "recovery-flow-fixture")
    result = import_verified(client, **kwargs, resume=True)
    assert result["mlflow_run_id"] == failed.info.run_id
    assert result["status"] == "resumed"
    assert result["remote_artifacts_verified"] == 1
    recovered = client.get_run(failed.info.run_id)
    assert recovered.info.status == "FINISHED"
    assert recovered.data.tags["prefect.flow_run_id"] == "recovery-flow-fixture"
    assert recovered.data.tags["artifact_verification"] == "verified"
    assert json.loads((run_dir / "tracking.json").read_text())["resumed"] is True
    events = [
        json.loads(line) for line in (run_dir / "tracking-events.jsonl").read_text().splitlines()
    ]
    assert [event["status"] for event in events] == ["uploading", "failed", "uploading", "resumed"]
    assert all(event["recorded_at"] for event in events)
    assert model.read_bytes() == b"immutable fixture model"
    assert len(client.search_runs([experiment])) == 1
    # Repeated completed imports verify the remote bytes too.
    with monkeypatch.context() as fault:
        fault.setattr(client, "download_artifacts", lambda *_args: str(corrupted))
        with pytest.raises(ValueError, match="Remote artifact checksum"):
            import_verified(client, **kwargs)
    # Ambiguous competing producers must never silently choose one remote run.
    client.create_run(
        experiment,
        tags={"logical_run_id": logical_id, "manifest_sha256": "unchanged-manifest"},
    )
    with pytest.raises(ValueError, match="duplicate"):
        import_verified(client, **kwargs, resume=True)
