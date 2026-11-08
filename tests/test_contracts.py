import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from iceberg.data import Preprocessing, RadarData, RadarDataset, augment, load_data, make_fixture
from iceberg.model import IcebergCNN
from iceberg.runtime import (
    evaluate,
    load_model,
    predict,
    probabilities,
    resolve_config,
    select_device,
    source_identity,
    train,
    verify_run,
)


def test_raw_bands_missing_angles_order_and_duplicate_rejection(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "train.json"
    records = json.loads(path.read_text())
    records[0]["band_1"] = [-25] * (75 * 75)
    path.write_text(json.dumps(records))
    data = load_data(path, labeled=True)
    assert data.ids[0] == "fixture-0"
    np.testing.assert_equal(data.images[0, 0], np.full((75, 75), -25))
    assert np.isnan(data.angles[0, 0])
    records[1]["id"] = records[0]["id"]
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Duplicate"):
        load_data(path, labeled=True)


def test_preprocessing_fits_only_training_and_survives_serialization():
    data = RadarData(
        ["a", "b", "holdout"],
        np.stack([np.full((2, 75, 75), value, dtype=np.float32) for value in [-20, -10, 1000]]),
        np.array([[30], [np.nan], [999]], dtype=np.float32),
        np.array([0, 1, 1], dtype=np.float32),
    )
    preprocessing = Preprocessing.fit(data, [0, 1])
    np.testing.assert_allclose(preprocessing.channel_mean, [-15, -15])
    np.testing.assert_allclose(preprocessing.channel_std, [5, 5])
    assert preprocessing.angle_median == 30
    data.images[2] = -9999
    data.angles[2] = -9999
    assert Preprocessing.fit(data, [0, 1]) == preprocessing
    restored = Preprocessing(**json.loads(json.dumps(preprocessing.to_dict())))
    np.testing.assert_allclose(restored.image(data.images[0]), -1)
    metadata = restored.metadata(data.angles)
    np.testing.assert_array_equal(metadata[:2], [[0, 0], [0, 1]])
    assert np.isfinite(metadata).all()
    data.angles[:] = np.nan
    assert np.isfinite(Preprocessing.fit(data, [0, 1]).metadata(data.angles)).all()


def test_spatial_transforms_preserve_channel_alignment_without_mutation():
    base = np.arange(75 * 75, dtype=np.float32).reshape(75, 75)
    image = np.stack([base, base + 100])
    original = image.copy()
    for seed in range(20):
        result = augment(image, np.random.default_rng(seed))
        np.testing.assert_allclose(result[1] - result[0], 100, atol=0.001)
        np.testing.assert_array_equal(image, original)
        assert result.shape == image.shape
        assert result.flags.c_contiguous
        assert np.isfinite(result).all()


def test_eval_batch_invariance_and_image_only_metadata_independence():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = IcebergCNN(filters=2, hidden=8, use_angle=False)
    images = torch.randn(5, 2, 75, 75)
    angles = torch.randn(5, 2)
    model.train()
    model(images, angles)  # Establish nontrivial native BN running statistics.
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        together = model(images, angles)
        singleton = torch.cat(
            [model(x[None], a[None]) for x, a in zip(images, angles, strict=True)]
        )
        changed_angle = model(images, angles * 1000)
    torch.testing.assert_close(together, singleton, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(together, changed_angle)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_tta_averages_probabilities_of_paired_flips():
    class CornerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, images, angles):
            return images[:, :, 0, 0].sum(1) * self.weight

    images = np.zeros((1, 2, 75, 75), dtype=np.float32)
    images[0, :, 0, 0] = 0.2
    images[0, :, 0, -1] = -0.3
    images[0, :, -1, 0] = 0.7
    images[0, :, -1, -1] = -1.1
    data = RadarData(["one"], images, np.array([[np.nan]]), None)
    preprocessing = Preprocessing([0, 0], [1, 1], 30, 30, 1)
    dataset = RadarDataset(data, [0], preprocessing)
    result = probabilities(CornerModel(), dataset, 1, tta="flips")
    expected = torch.sigmoid(torch.tensor([0.4, -0.6, 1.4, -2.2])).mean()
    np.testing.assert_allclose(result, [float(expected)], atol=1e-7)


def test_invalid_configuration_and_unavailable_cuda_are_explicit(monkeypatch):
    with pytest.raises(ValueError, match="Unknown"):
        resolve_config({"epoch_repeats": 8})
    with pytest.raises(ValueError, match="finite"):
        resolve_config({"learning_rate": float("nan")})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="unavailable"):
        select_device("cuda")


def test_train_reload_holdout_submission_and_bundle_contract(tmp_path):
    data_dir = tmp_path / "data"
    make_fixture(data_dir)
    config = json.loads(Path("configs/smoke.json").read_text())
    run = tmp_path / "run"
    manifest = train(data_dir / "train.json", run, config, fixture=True)
    assert manifest["status"] == "completed"
    assert manifest["run_kind"] == "synthetic_fixture"
    assert manifest["training_examples_seen"] == 24
    assert manifest["optimizer_steps"] == 3
    assert verify_run(run, data_dir / "train.json") == manifest
    split = list(csv.DictReader((run / "split.csv").open()))
    assert len({r["id"] for r in split}) == len(split)
    assert {r["role"] for r in split} == {"train", "validation"}
    _, saved = load_model(run / "model.pt", device="cpu")
    assert saved["split_sha256"] == manifest["split_sha256"]
    assert saved["preprocessing"] == manifest["preprocessing"]
    data = load_data(data_dir / "train.json", labeled=True)
    fit_indices = [i for i, row in enumerate(split) if row["role"] == "train"]
    fit_statistics = Preprocessing.fit(data, fit_indices).to_dict()
    for key, value in saved["preprocessing"].items():
        np.testing.assert_allclose(value, fit_statistics[key], rtol=1e-12, atol=1e-12)
    expected = json.loads((run / "metrics.json").read_text())
    assert evaluate(run / "model.pt", data_dir / "train.json", device="cpu") == expected
    output = tmp_path / "submission.csv"
    receipt = predict(run / "model.pt", data_dir / "test.json", output, tta="flips")
    rows = list(csv.DictReader(output.open()))
    assert receipt["rows"] == 8
    assert receipt["tta"] == "flips"
    assert [r["id"] for r in rows] == [f"test-fixture-{i}" for i in range(8)]
    assert all(0 <= float(r["is_iceberg"]) <= 1 for r in rows)
    altered = tmp_path / "changed.json"
    altered.write_text((data_dir / "train.json").read_text() + " ")
    with pytest.raises(ValueError, match="differs"):
        evaluate(run / "model.pt", altered)
    with pytest.raises(ValueError, match="differs"):
        verify_run(run, altered)
    (run / "metrics.json").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_run(run)


def test_source_fingerprint_is_identical_in_an_installed_package_layout(tmp_path, monkeypatch):
    import iceberg.runtime as runtime

    original = source_identity()["python_source_sha256"]
    installed = tmp_path / "site-packages" / "iceberg"
    shutil.copytree(
        Path(runtime.__file__).parent, installed, ignore=shutil.ignore_patterns("__pycache__")
    )
    monkeypatch.setattr(runtime, "__file__", str(installed / "runtime.py"))
    identity = source_identity()
    assert identity["python_source_sha256"] == original
    assert identity["git_commit"] is None


def test_source_fingerprint_survives_a_runtime_without_git(monkeypatch):
    import iceberg.runtime as runtime

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git is not installed")

    monkeypatch.setattr(runtime.subprocess, "run", missing_git)
    identity = source_identity()
    assert identity["git_commit"] is None
    assert identity["git_dirty"] is None
    assert len(identity["python_source_sha256"]) == 64
