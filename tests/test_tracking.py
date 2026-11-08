import json
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")

from iceberg.data import make_fixture  # noqa: E402
from iceberg.runtime import train  # noqa: E402
from iceberg.tracking import track  # noqa: E402


def test_sqlite_import_is_idempotent_and_rejects_conflicting_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    data = tmp_path / "data"
    make_fixture(data)
    run_dir = tmp_path / "run"
    config = json.loads(Path("configs/smoke.json").read_text())
    train(data / "train.json", run_dir, config, fixture=True)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["git_commit"] = None
    manifest["code_bundle"] = {
        "git_commit": "f" * 40,
        "python_source_sha256": manifest["source"]["python_source_sha256"],
    }
    manifest_path.write_text(json.dumps(manifest))
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    previous_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        experiment_id = client.create_experiment(
            "kaggle/iceberg",
            artifact_location=(tmp_path / "artifacts").as_uri(),
        )
        first = track(run_dir, tracking_uri)
        repeat = track(run_dir, tracking_uri)
        assert repeat["status"] == "already_imported"
        assert first["mlflow_run_id"] == repeat["mlflow_run_id"]
        assert client.get_run(first["mlflow_run_id"]).data.tags["source_commit"] == "f" * 40
        assert len(client.search_runs([experiment_id])) == 1
        retrieved = Path(client.download_artifacts(first["mlflow_run_id"], "model.pt"))
        assert retrieved.read_bytes() == (run_dir / "model.pt").read_bytes()
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["profile"] = "changed-after-import"
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="Conflicting"):
            track(run_dir, tracking_uri)
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_cv_import_links_all_folds_and_is_idempotent(tmp_path, monkeypatch):
    from iceberg.cv import cross_validate
    from iceberg.tracking import track_cv

    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    data = tmp_path / "data"
    make_fixture(data)
    config = json.loads(Path("configs/smoke.json").read_text())
    run_dir = tmp_path / "cv"
    cross_validate(data / "train.json", run_dir, config, folds=2, seeds=[17, 27], fixture=True)
    uri = f"sqlite:///{tmp_path / 'cv-tracking.db'}"
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        client = mlflow.MlflowClient()
        experiment = client.create_experiment(
            "kaggle/iceberg", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        first = track_cv(run_dir, uri)
        repeat = track_cv(run_dir, uri)
        assert repeat["status"] == "already_imported"
        assert repeat["mlflow_run_id"] == first["mlflow_run_id"]
        assert len(first["fold_runs"]) == 4
        runs = client.search_runs([experiment])
        assert len(runs) == 5
        children = [
            r for r in runs if r.data.tags.get("mlflow.parentRunId") == first["mlflow_run_id"]
        ]
        assert len(children) == 4
        expected = json.loads((run_dir / "analysis.json").read_text())["pooled_oof"]["log_loss"]
        parent = client.get_run(first["mlflow_run_id"])
        assert parent.info.status == "FINISHED"
        assert parent.data.metrics["oof.log_loss"] == expected
        downloaded = Path(client.download_artifacts(first["mlflow_run_id"], "analysis.json"))
        assert downloaded.read_bytes() == (run_dir / "analysis.json").read_bytes()
    finally:
        mlflow.set_tracking_uri(previous)
