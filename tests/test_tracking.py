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
