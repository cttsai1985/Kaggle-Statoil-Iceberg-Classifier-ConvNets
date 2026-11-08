import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from iceberg.data import augment, load_data, make_fixture, spatial_affine
from iceberg.model import LegacyBatchNorm
from iceberg.runtime import evaluate, predict, train


def test_scaling_sentinel_order_and_duplicate_rejection(tmp_path):
    make_fixture(tmp_path)
    records = json.loads((tmp_path / "train.json").read_text())
    records[0]["band_1"] = [-25] * (75 * 75)
    (tmp_path / "train.json").write_text(json.dumps(records))
    data = load_data(tmp_path / "train.json", labeled=True)
    assert data.ids[0] == "fixture-0"
    np.testing.assert_equal(data.images[0, 0], np.full((75, 75), 0.25))
    assert data.angles[0, 0] == -1
    records[1]["id"] = records[0]["id"]
    (tmp_path / "train.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Duplicate"):
        load_data(tmp_path / "train.json", labeled=True)


def test_spatial_transforms_preserve_channel_alignment():
    base = np.arange(75 * 75, dtype=np.float32).reshape(75, 75)
    image = np.stack([base, base + 100])
    transformed = spatial_affine(image, np.array([[0.9, -0.1], [0.1, 0.9]]))
    np.testing.assert_allclose(transformed[1] - transformed[0], 100, atol=0.001)
    for seed in range(20):
        result = augment(image, np.random.default_rng(seed))
        np.testing.assert_allclose(result[1] - result[0], 100, atol=0.001)
        assert result.flags.c_contiguous


def test_legacy_batch_norm_uses_population_variance_and_last_batch():
    layer = LegacyBatchNorm(1)
    layer(torch.tensor([[1.0], [3.0]]))
    torch.testing.assert_close(layer.running_mean, torch.tensor([2.0]))
    torch.testing.assert_close(layer.running_var, torch.tensor([1.0]))
    layer(torch.tensor([[8.0]]))
    torch.testing.assert_close(layer.running_mean, torch.tensor([8.0]))
    layer.eval()
    assert layer(torch.tensor([[8.0]])).item() == 0.0


def test_train_reload_holdout_and_submission_contract(tmp_path):
    data_dir = tmp_path / "data"
    make_fixture(data_dir)
    config = json.loads(Path("configs/smoke.json").read_text())
    run = tmp_path / "run"
    manifest = train(data_dir / "train.json", run, config, fixture=True)
    assert manifest["status"] == "completed"
    assert manifest["run_kind"] == "synthetic_fixture"
    split = list(csv.DictReader((run / "split.csv").open()))
    assert len({r["id"] for r in split}) == len(split)
    assert {r["role"] for r in split} == {"train", "validation"}
    expected = json.loads((run / "metrics.json").read_text())
    assert evaluate(run / "model.pt", data_dir / "train.json") == expected
    output = tmp_path / "submission.csv"
    receipt = predict(run / "model.pt", data_dir / "test.json", output)
    rows = list(csv.DictReader(output.open()))
    assert receipt["rows"] == 8
    assert [r["id"] for r in rows] == [f"test-fixture-{i}" for i in range(8)]
    assert all(0 <= float(r["is_iceberg"]) <= 1 for r in rows)
    altered = tmp_path / "changed.json"
    altered.write_text((data_dir / "train.json").read_text() + " ")
    with pytest.raises(ValueError, match="differs"):
        evaluate(run / "model.pt", altered)
