"""Independently reconcile ablation log loss against raw labels and retained OOF rows."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def loss(targets, probabilities):
    probability = np.clip(
        np.asarray(probabilities, dtype=float), np.finfo(float).eps, 1 - np.finfo(float).eps
    )
    target = np.asarray(targets, dtype=float)
    return float(np.mean(-target * np.log(probability) - (1 - target) * np.log1p(-probability)))


def verify(metadata_run, image_run, data_path, comparison):
    study = json.loads((comparison / "study.json").read_text())
    raw = {row["id"]: row for row in json.loads(data_path.read_text())}
    seeds = study["protocol"]["model_seeds"]
    folds = {row["id"]: int(row["fold"]) for row in read_rows(metadata_run / "folds.csv")}
    assert set(folds) == set(raw)
    known_ids = {
        identifier for identifier, row in raw.items() if row["inc_angle"] not in ("na", None)
    }
    paths = {
        "image_only_flips": image_run / "oof.csv",
        "image_and_angle_flips": metadata_run / "oof.csv",
        "image_only_none": comparison / "image_only-no-tta-oof.csv",
        "image_and_angle_none": comparison / "image_and_angle-no-tta-oof.csv",
    }
    checks = []
    recomputed = {}
    for name, path in paths.items():
        rows = read_rows(path)
        assert len(rows) == len(raw) * len(seeds)
        predictions = {}
        for row in rows:
            key = (int(row["seed"]), row["id"])
            assert key not in predictions and key[0] in seeds and key[1] in raw
            assert int(row["target"]) == raw[key[1]]["is_iceberg"]
            assert int(row["angle_missing"]) == (key[1] not in known_ids)
            assert int(row["fold"]) == folds[key[1]]
            predictions[key] = float(row["probability"])
        assert set(predictions) == {(seed, identifier) for seed in seeds for identifier in raw}
        assert all(np.isfinite(p) and 0 <= p <= 1 for p in predictions.values())
        variant = study["variants"][name]

        def check(ids, seed, expected, label, predictions=predictions, name=name):
            ids = sorted(ids)
            predicted = [
                predictions[(seed, identifier)]
                if seed is not None
                else np.mean([predictions[(item, identifier)] for item in seeds])
                for identifier in ids
            ]
            actual = loss([raw[identifier]["is_iceberg"] for identifier in ids], predicted)
            error = abs(actual - expected)
            assert error < 1e-12, (name, label, actual, expected)
            checks.append(
                {"condition": name, "scope": label, "rows": len(ids), "absolute_error": error}
            )
            return actual

        check(raw, None, variant["pooled_oof"]["log_loss"], "all-row ensemble")
        recomputed[name] = check(
            known_ids, None, variant["strata"]["known_angle"]["log_loss"], "known-angle ensemble"
        )
        for row in variant["seed_results"]:
            check(
                known_ids,
                row["seed"],
                row["strata"]["known_angle"]["log_loss"],
                f"known-angle seed {row['seed']}",
            )
        for row in variant["seed_fold_metrics"]:
            ids = {identifier for identifier in known_ids if folds[identifier] == row["fold"]}
            check(
                ids,
                row["seed"],
                row["known_angle_log_loss"],
                f"known-angle seed {row['seed']} fold {row['fold']}",
            )
        for row in variant["known_angle_fold_metrics"]:
            ids = {identifier for identifier in known_ids if folds[identifier] == row["fold"]}
            check(
                ids, None, row["known_angle_log_loss"], f"known-angle ensemble fold {row['fold']}"
            )
    for pair in study["paired_comparisons"]:
        assert (
            abs(
                recomputed[pair["after"]]
                - recomputed[pair["before"]]
                - pair["known_angle_log_loss_delta"]
            )
            < 1e-12
        )
    return {
        "status": "passed",
        "method": (
            "Independent NumPy binary cross-entropy from private OOF CSVs, joined to raw "
            "JSON targets, missingness and frozen fold IDs; not the package metric helper"
        ),
        "rows_per_seed": len(raw),
        "known_angle_rows": len(known_ids),
        "missing_angle_rows": len(raw) - len(known_ids),
        "comparisons_checked": len(study["paired_comparisons"]),
        "metric_reconciliations": len(checks),
        "maximum_absolute_error": max(check["absolute_error"] for check in checks),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata-run", "image-run", "data", "comparison", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.metadata_run, args.image_run, args.data, args.comparison)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
