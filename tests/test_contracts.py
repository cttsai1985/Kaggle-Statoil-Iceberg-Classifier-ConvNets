import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from iceberg.data import augment, load_data, make_fixture, spatial_affine
from iceberg.model import LegacyBatchNorm
from iceberg.runtime import (
    evaluate,
    legacy_adam_step,
    predict,
    steps_per_epoch,
    train,
    training_batches,
)


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


def test_affine_matches_historical_nearest_neighbor_interpolation():
    base = np.arange(16, dtype=np.float32).reshape(4, 4)
    image = np.stack([base, base + 100])
    # Center=2.5 and zoom=.5 map coordinates to [1.25, 1.75, 2.25, 2.75].
    expected = base[np.ix_([1, 2, 2, 3], [1, 2, 2, 3])]
    np.testing.assert_array_equal(
        spatial_affine(image, np.eye(2) * 0.5), np.stack([expected, expected + 100])
    )


def test_reference_generator_keeps_its_position_across_epochs():
    loader = torch.utils.data.DataLoader(range(1348), batch_size=64)
    stream = training_batches(loader)
    steps = steps_per_epoch(1348, 64, 8)
    assert steps == 169
    last_batches, total = [], 0
    for _ in range(3):
        for _ in range(steps):
            batch = next(stream)
            total += len(batch)
        last_batches.append(batch)
    assert [len(batch) for batch in last_batches] == [64, 64, 64]
    assert [int(batch[0]) for batch in last_batches] == [896, 448, 0]
    assert total == 31068


def test_spatial_batch_norm_matches_population_moments():
    values = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    layer = LegacyBatchNorm(2)
    expected_mean = torch.tensor([5.5, 9.5])
    expected_variance = torch.tensor([17.25, 17.25])
    expected = (values - expected_mean[None, :, None, None]) / torch.sqrt(
        expected_variance[None, :, None, None] + 0.001
    )
    torch.testing.assert_close(layer(values), expected)
    torch.testing.assert_close(layer.running_mean, expected_mean)
    torch.testing.assert_close(layer.running_var, expected_variance)
    layer.eval()
    torch.testing.assert_close(layer(values), expected)


def test_adam_matches_historical_equation_including_small_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.0005, eps=1e-8)
    expected = np.array([1.0, -1.0])
    first_moment, second_moment = np.zeros(2), np.zeros(2)
    for step, gradient in enumerate([[0.1, 1e-10], [-0.2, -2e-10], [0.03, 3e-10]], 1):
        gradient = np.array(gradient)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        rate = 0.0005 * np.sqrt(1 - 0.999**step) / (1 - 0.9**step)
        expected -= rate * first_moment / (np.sqrt(second_moment) + 1e-8)
        parameter.grad = torch.from_numpy(gradient)
        legacy_adam_step(optimizer, step - 1)
        np.testing.assert_allclose(parameter.detach().numpy(), expected, atol=1e-14, rtol=0)


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
