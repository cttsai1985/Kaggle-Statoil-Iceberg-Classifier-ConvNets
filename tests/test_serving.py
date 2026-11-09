"""Prediction equivalence and immutable provenance at the serving boundary."""

import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
import torch

from iceberg.data import Preprocessing, RadarDataset, sha256
from iceberg.runtime import ARTIFACTS, build_model, load_model, probabilities, resolve_config
from iceberg.serving import export_model, frame_from_records, radar_from_frame


def native_bundle(path: Path) -> Path:
    """Write a small native-format, untrained fixture; no benchmark claim is made."""
    path.mkdir()
    torch.manual_seed(25)
    config = resolve_config(
        {"filters": 2, "hidden": 8, "threads": 1, "use_angle": True, "tta": "flips"}
    )
    preprocessing = Preprocessing([0.0, 0.0], [1.0, 1.0], 35.0, 35.0, 5.0)
    torch.save(
        {
            "schema_version": 2,
            "model": build_model(config).state_dict(),
            "config": config,
            "preprocessing": preprocessing.to_dict(),
        },
        path / "model.pt",
    )
    (path / "config.json").write_text(json.dumps(config))
    (path / "split.csv").write_text("id,role\nfixture,validation\n")
    for name in ARTIFACTS - {"model.pt", "config.json", "split.csv"}:
        (path / name).write_text("{}\n")
    manifest = {
        "schema_version": 2,
        "status": "completed",
        "run_kind": "synthetic_fixture",
        "logical_run_id": "serving-fixture",
        "source": {"fixture": True},
        "config": config,
        "config_sha256": sha256(path / "config.json"),
        "split_sha256": sha256(path / "split.csv"),
        "artifacts": {name: sha256(path / name) for name in ARTIFACTS},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def radar_frame():
    return frame_from_records(
        [
            {"band_1": [0.1] * 5625, "band_2": [-0.2] * 5625, "inc_angle": 37.5},
            {"band_1": [0.3] * 5625, "band_2": [-0.5] * 5625, "inc_angle": "na"},
        ]
    )


def test_package_predictions_match_native_preprocessing_and_saved_flip_recipe(tmp_path):
    run = native_bundle(tmp_path / "run")
    frame = radar_frame()
    checkpoint_before = sha256(run / "model.pt")
    manifest_before = sha256(run / "manifest.json")
    model, saved = load_model(run / "model.pt", device="cpu")
    data = radar_from_frame(frame, 4)
    expected = probabilities(
        model,
        RadarDataset(data, range(len(data.ids)), Preprocessing(**saved["preprocessing"])),
        batch_size=64,
        tta=saved["config"]["tta"],
    )
    package = tmp_path / "package"
    receipt = export_model(run, package, threads=1, max_batch_rows=4)
    loaded = mlflow.pyfunc.load_model(str(package))
    actual = loaded.predict(frame)
    np.testing.assert_allclose(actual.is_iceberg, expected, atol=1e-7, rtol=0)
    assert list(actual) == ["is_iceberg"]
    assert receipt["inference"]["tta"] == "flips"
    assert receipt["producer_run_kind"] == "synthetic_fixture"
    assert sha256(run / "model.pt") == checkpoint_before
    assert sha256(run / "manifest.json") == manifest_before
    assert not list(package.rglob("validation.csv"))
    assert not list(package.rglob("split.csv"))
    with pytest.raises(ValueError, match="immutable"):
        export_model(run, package)
    copied = next((package / "artifacts").rglob("model.pt"))
    copied.chmod(0o644)
    copied.write_bytes(copied.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="producer artifacts"):
        mlflow.pyfunc.load_model(str(package))


def test_input_contract_rejects_malformed_or_unbounded_payloads():
    frame = radar_frame()
    assert np.isnan(radar_from_frame(frame, 2).angles[1, 0])
    assert np.isnan(radar_from_frame(frame.drop(columns="inc_angle"), 2).angles).all()
    with pytest.raises(ValueError, match="between 1 and 1"):
        radar_from_frame(frame, 1)
    bad = frame.copy(deep=True)
    bad.at[0, "band_1"] = [0.0] * 5624
    with pytest.raises(ValueError, match="numeric arrays|5625"):
        radar_from_frame(bad, 2)
    bad = frame.copy(deep=True)
    bad.at[0, "band_1"] = [float("inf")] * 5625
    with pytest.raises(ValueError, match="finite"):
        radar_from_frame(bad, 2)
    with pytest.raises(ValueError, match="columns"):
        radar_from_frame(pd.DataFrame({"wrong": [1.0]}), 2)


def test_export_rejects_changed_producer_before_creating_package(tmp_path):
    run = native_bundle(tmp_path / "run")
    (run / "model.pt").write_bytes(b"changed checkpoint")
    output = tmp_path / "package"
    with pytest.raises(ValueError, match="checksum"):
        export_model(run, output)
    assert not output.exists()
