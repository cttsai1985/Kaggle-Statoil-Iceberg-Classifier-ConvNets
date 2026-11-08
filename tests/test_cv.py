import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import log_loss

from iceberg.cv import (
    acquisition_groups,
    analyze_cv,
    analyze_predictions,
    build_folds,
    cross_validate,
    predict_cv,
    read_csv,
)
from iceberg.data import Preprocessing, load_data, make_fixture, sha256
from iceberg.runtime import load_model, predict, validate_split_indices


def test_stratified_folds_cover_every_row_and_group_audit_is_disjoint(tmp_path):
    make_fixture(tmp_path, rows=40)
    data = load_data(tmp_path / "train.json", labeled=True)
    # Repeated acquisitions contain both classes; missing angles are not one giant group.
    data.angles[:, 0] = np.repeat(np.arange(10, dtype=float), 4)
    data.angles[:4, 0] = np.nan
    groups = acquisition_groups(data)
    assert len(set(groups[:4])) == 4
    for strategy in ["stratified", "grouped"]:
        splits = build_folds(data, folds=5, strategy=strategy, split_seed=25)
        again = build_folds(data, folds=5, strategy=strategy, split_seed=25)
        coverage = np.zeros(40, dtype=int)
        for (fit, validation), (_, repeated) in zip(splits, again, strict=True):
            np.testing.assert_array_equal(validation, repeated)
            coverage[validation] += 1
            assert len(np.unique(data.targets[validation])) == 2
            if strategy == "grouped":
                assert not set(groups[fit]) & set(groups[validation])
            else:
                assert data.targets[validation].mean() == 0.5
        np.testing.assert_array_equal(coverage, 1)


@pytest.mark.parametrize(
    "parts",
    [
        ([0, 1], [1, 2, 3]),
        ([0, 0], [1, 2, 3]),
        ([0, 1], [2]),
        ([0.0, 1.0], [2, 3]),
        ([0, 1], [2, 4]),
        ([], [0, 1, 2, 3]),
    ],
)
def test_explicit_splits_reject_overlap_missing_duplicate_or_invalid_rows(parts):
    with pytest.raises(ValueError):
        validate_split_indices(4, parts)


def test_seed_pairing_rejects_misaligned_oof():
    records = [
        {
            "seed": str(seed),
            "id": str(i),
            "fold": str(i % 2),
            "target": str(i % 2),
            "probability": str(0.2 + 0.5 * (i % 2)),
            "inc_angle": "30",
            "angle_missing": "0",
            "group": "angle:30",
        }
        for seed in [1017, 2027]
        for i in range(4)
    ]
    analysis, _, ensemble = analyze_predictions(records, [1017, 2027])
    assert analysis["rows"] == len(ensemble) == 4
    assert analysis["seed_distribution"]["log_loss"]["range"] == 0
    records[-1]["fold"] = "0"
    with pytest.raises(ValueError, match="same OOF"):
        analyze_predictions(records, [1017, 2027])


def test_real_training_cv_fixture_keeps_last_epoch_and_train_only_transforms(tmp_path):
    data_dir = tmp_path / "data"
    make_fixture(data_dir)
    data_path = data_dir / "train.json"
    config = json.loads(Path("configs/smoke.json").read_text()) | {"epochs": 2}
    run = tmp_path / "cv"
    manifest = cross_validate(data_path, run, config, folds=2, seeds=[1017, 2027], fixture=True)
    assert manifest["status"] == "completed"
    assert manifest["protocol"]["checkpoint_selection"] == "last"
    assert len(manifest["fold_runs"]) == 4
    assert manifest["grouped_model_evaluation"] == "not_run"
    assert all(item["shared_known_angles"] == 0 for item in manifest["grouped_split_audit"])
    data = load_data(data_path, labeled=True)
    for record in manifest["fold_runs"]:
        child = run / record["directory"]
        _, checkpoint = load_model(child / "model.pt", device="cpu")
        assert checkpoint["epoch"] == 2
        assert checkpoint["checkpoint_selection"] == "last"
        splits = read_csv(child / "split.csv")
        fit = [i for i, row in enumerate(splits) if row["role"] == "train"]
        expected = Preprocessing.fit(data, fit).to_dict()
        for name, value in expected.items():
            np.testing.assert_allclose(checkpoint["preprocessing"][name], value)
    analysis = analyze_cv(run, data_path)
    oof = read_csv(run / "oof.csv")
    ensemble = read_csv(run / "ensemble-oof.csv")
    assert len(oof) == 64
    assert len(ensemble) == analysis["rows"] == 32
    assert len({(row["seed"], row["id"]) for row in oof}) == 64
    assert [record["rows"] for record in analysis["seed_results"]] == [32, 32]
    assert analysis["pooled_oof"]["log_loss"] == pytest.approx(
        log_loss(
            [int(row["target"]) for row in ensemble],
            [float(row["probability"]) for row in ensemble],
        )
    )
    assert sum(item["rows"] for item in analysis["calibration_bins"]) == 32
    errors = read_csv(run / "errors.csv")
    assert [float(row["row_log_loss"]) for row in errors] == sorted(
        [float(row["row_log_loss"]) for row in errors], reverse=True
    )
    # A valid parent checksum cannot hide corruption in a retained child model.
    checkpoint_path = run / manifest["fold_runs"][0]["directory"] / "model.pt"
    with checkpoint_path.open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        analyze_cv(run)


def test_cv_inference_uses_every_model_and_preserves_input_order(tmp_path, monkeypatch):
    import iceberg.cv as cv

    data_dir = tmp_path / "data"
    make_fixture(data_dir)
    test_path = data_dir / "test.json"
    test_path.write_text(json.dumps(list(reversed(json.loads(test_path.read_text())))))
    config = json.loads(Path("configs/smoke.json").read_text()) | {"tta": "flips"}
    run = tmp_path / "cv"
    manifest = cross_validate(
        data_dir / "train.json", run, config, folds=2, seeds=[1017, 2027], fixture=True
    )
    calls = []

    def counted_load(path, *, labeled):
        calls.append(path)
        return load_data(path, labeled=labeled)

    monkeypatch.setattr(cv, "load_data", counted_load)
    # A later consumer can load an older verified producer without claiming identical source.
    consumer_source = {**manifest["source"], "python_source_sha256": "new-consumer-source"}
    monkeypatch.setattr(cv, "source_identity", lambda: consumer_source)
    output = tmp_path / "submission.csv"
    receipt = predict_cv(run, test_path, output, device="cpu")
    assert calls == [test_path]
    assert receipt["model_count"] == len(manifest["fold_runs"]) == 4
    assert receipt["training_source"] == manifest["source"]
    assert receipt["inference_source"] == consumer_source
    assert receipt["cv_manifest_sha256"] == sha256(run / "cv-manifest.json")
    assert receipt["input_sha256"] == sha256(test_path)
    assert receipt["output_sha256"] == sha256(output)
    assert [item["sha256"] for item in receipt["checkpoints"]] == [
        item["checkpoint_sha256"] for item in manifest["fold_runs"]
    ]
    assert json.loads(output.with_suffix(".receipt.json").read_text()) == receipt
    rows = read_csv(output)
    assert [row["id"] for row in rows] == [row["id"] for row in json.loads(test_path.read_text())]
    assert len(rows) == receipt["rows"] == 8
    individual = []
    for i, record in enumerate(manifest["fold_runs"]):
        path = tmp_path / f"individual-{i}.csv"
        predict(run / record["directory"] / "model.pt", test_path, path, device="cpu")
        individual.append([float(row["is_iceberg"]) for row in read_csv(path)])
    np.testing.assert_allclose(
        [float(row["is_iceberg"]) for row in rows], np.mean(individual, axis=0), atol=1e-7
    )
    with pytest.raises(ValueError, match="overlap"):
        predict_cv(run, data_dir / "train.json", tmp_path / "invalid.csv", device="cpu")
    assert not (tmp_path / "invalid.csv").exists()
    with pytest.raises(ValueError, match="overwrite"):
        predict_cv(run, test_path, run / "oof.csv", device="cpu")
    with pytest.raises(ValueError, match="input differs"):
        analyze_cv(run, test_path)
    manifest["source"]["python_source_sha256"] = "tampered-producer-source"
    (run / "cv-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source fingerprint"):
        predict_cv(run, test_path, tmp_path / "invalid.csv", device="cpu")
