"""Mine retained grouped OOF errors without refitting or changing an evaluation population."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

COHORT_FRACTION = 0.10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verified_file(root, name, expected):
    require(Path(name).name == name, "Artifact name must be a basename")
    path = root / name
    require(path.is_file() and not path.is_symlink(), "Expected a regular artifact")
    require(digest(path) == expected, f"Artifact checksum differs: {name}")
    return path


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_metadata(path):
    """Stream the official JSON; retain labels/angles, never the radar pixel arrays."""
    decoder = json.JSONDecoder()
    metadata = {}
    with path.open() as stream:
        buffer = stream.read(262144).lstrip()
        require(buffer.startswith("["), "Expected a JSON record array")
        buffer = buffer[1:]
        first = True
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                buffer = stream.read(262144)
                require(bool(buffer), "Incomplete JSON record array")
                continue
            if buffer.startswith("]"):
                require(not (buffer[1:] + stream.read()).strip(), "Unexpected JSON suffix")
                break
            if not first:
                require(buffer.startswith(","), "Missing JSON record separator")
                buffer = buffer[1:].lstrip()
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    chunk = stream.read(262144)
                    require(bool(chunk), "Incomplete JSON record")
                    buffer += chunk
            buffer = buffer[end:]
            first = False
            identifier = row["id"]
            require(identifier not in metadata, "Duplicate raw ID")
            target = row["is_iceberg"]
            require(target in (0, 1), "Raw target must be binary")
            angle = row["inc_angle"]
            angle = np.nan if angle in (None, "na") else float(angle)
            require(not np.isinf(angle), "Invalid raw angle")
            metadata[identifier] = (int(target), angle)
    require(bool(metadata), "Empty raw population")
    return metadata


def row_losses(target, probability):
    target, probability = np.asarray(target, dtype=float), np.asarray(probability, dtype=float)
    require(
        target.ndim == 1 and target.shape == probability.shape and len(target), "Invalid vectors"
    )
    require(np.isin(target, [0, 1]).all(), "Targets must be binary")
    require(
        np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
        "Invalid probabilities",
    )
    eps = np.finfo(float).eps
    probability = np.clip(probability, eps, 1 - eps)
    return -(target * np.log(probability) + (1 - target) * np.log1p(-probability))


def hardest(loss):
    count = max(1, math.ceil(COHORT_FRACTION * len(loss)))
    mask = np.zeros(len(loss), dtype=bool)
    mask[np.argsort(-loss, kind="stable")[:count]] = True
    return mask


def summarize(target, probability, folds, angles, seeds):
    loss = row_losses(target, probability)
    target, folds, angles = np.asarray(target), np.asarray(folds), np.asarray(angles, dtype=float)
    require(target.shape == folds.shape == angles.shape, "Population lengths differ")
    require(
        np.isfinite(folds).all() and np.equal(folds, np.floor(folds)).all(), "Invalid fold values"
    )
    require(not np.isinf(angles).any(), "Angles must be finite or missing")
    require((folds >= 0).all(), "Fold values must be nonnegative")
    hard = hardest(loss)
    total_loss = float(loss.sum())

    def cohort(mask):
        count = int(mask.sum())
        return {
            "rows": count,
            "iceberg_rows": int((mask & (target == 1)).sum()),
            "mean_log_loss": float(loss[mask].mean()) if count else None,
            "hard_rows": int((mask & hard).sum()),
            "hard_rate": float(hard[mask].mean()) if count else None,
            "share_of_total_loss": float(loss[mask].sum() / total_loss) if total_loss else 0.0,
        }

    strata = {
        "known_angle": np.isfinite(angles),
        "missing_angle": np.isnan(angles),
        "ship": target == 0,
        "iceberg": target == 1,
        "known_angle_below_35": np.isfinite(angles) & (angles < 35),
        "known_angle_35_to_below_40": np.isfinite(angles) & (angles >= 35) & (angles < 40),
        "known_angle_at_least_40": np.isfinite(angles) & (angles >= 40),
    }
    seed_masks = {}
    for seed, prediction in seeds.items():
        require(np.asarray(prediction).shape == target.shape, "Seed population differs")
        seed_masks[str(seed)] = hardest(row_losses(target, prediction))
    require(len(seed_masks) == 2, "This diagnostic requires exactly two declared seeds")
    left, right = seed_masks.values()
    return {
        "rows": len(loss),
        "pooled_log_loss": float(loss.mean()),
        "cohort_rule": {
            "loss": "binary cross-entropy",
            "fraction": COHORT_FRACTION,
            "count_rule": "ceiling",
            "tie_break": "retained OOF order",
        },
        "hard_cohort": {
            **cohort(hard),
            "minimum_selected_row_loss": float(loss[hard].min()),
            "ship_rows": int((hard & (target == 0)).sum()),
            "iceberg_rows": int((hard & (target == 1)).sum()),
            "known_angle_rows": int((hard & np.isfinite(angles)).sum()),
        },
        "strata": {name: cohort(mask) for name, mask in strata.items()},
        "folds": [{"fold": int(fold), **cohort(folds == fold)} for fold in np.unique(folds)],
        "seed_error_overlap": {
            "seed_ids": list(seed_masks),
            "hard_rows_per_seed": int(left.sum()),
            "intersection_rows": int((left & right).sum()),
            "union_rows": int((left | right).sum()),
            "jaccard": float((left & right).sum() / (left | right).sum()),
            "ensemble_hard_rows_in_both_seed_cohorts": int((hard & left & right).sum()),
        },
    }


def validate_population(rows, seed_rows, assignments, metadata, protocol):
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)) and set(ids) == set(metadata), "OOF ID population differs")
    lookup = {row["id"]: row for row in assignments}
    require(
        len(lookup) == len(assignments) and set(lookup) == set(ids), "Fold ID population differs"
    )
    declared_seeds = protocol["model_seeds"]
    require(len(declared_seeds) == len(set(declared_seeds)) == 2, "Require two unique model seeds")
    require(
        set(int(row["seed"]) for row in seed_rows) == set(declared_seeds), "Seed identities differ"
    )
    grouped = {seed: {} for seed in declared_seeds}
    for row in seed_rows:
        seed, identifier = int(row["seed"]), row["id"]
        require(identifier not in grouped[seed], "Duplicate seed/ID OOF prediction")
        grouped[seed][identifier] = row
    for seed in grouped:
        require(set(grouped[seed]) == set(ids), "Seed OOF population differs")
    probabilities = {seed: [] for seed in declared_seeds}
    targets, folds, angles, predicted = [], [], [], []
    group_folds = {}
    for row in rows:
        identifier = row["id"]
        target, angle = metadata[identifier]
        fold = int(row["fold"])
        missing = np.isnan(angle)
        group = f"missing:{identifier}" if missing else f"angle:{float(angle).hex()}"
        require(0 <= fold < protocol["folds"], "Fold is outside declared protocol")
        require(int(row["target"]) == target, "OOF label differs from raw input")
        require(int(row["angle_missing"]) == int(missing), "OOF missing-angle flag differs")
        observed_angle = np.nan if row["inc_angle"] == "" else float(row["inc_angle"])
        require(
            (missing and np.isnan(observed_angle)) or observed_angle == angle, "OOF angle differs"
        )
        require(row["group"] == group == lookup[identifier]["group"], "Acquisition proxy differs")
        require(fold == int(lookup[identifier]["fold"]), "OOF fold differs from saved assignment")
        require(group_folds.setdefault(group, fold) == fold, "Acquisition group crosses folds")
        values = []
        for seed in declared_seeds:
            item = grouped[seed][identifier]
            for key in ["fold", "target", "inc_angle", "angle_missing", "group"]:
                require(item[key] == row[key], "Seed metadata differs from ensemble")
            value = float(item["probability"])
            values.append(value)
            probabilities[seed].append(value)
        prediction = float(row["probability"])
        require(
            np.isclose(prediction, np.mean(values), rtol=0, atol=1e-12),
            "Ensemble differs from seed mean",
        )
        targets.append(target)
        folds.append(fold)
        angles.append(angle)
        predicted.append(prediction)
    require(set(folds) == set(range(protocol["folds"])), "Missing declared fold")
    return targets, predicted, folds, angles, probabilities


def analyze(run_dir, data_path):
    run_dir, data_path = Path(run_dir), Path(data_path)
    manifest_path = run_dir / "cv-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("project_id") == "iceberg" and manifest.get("status") == "completed",
        "Expected completed Iceberg CV",
    )
    require(manifest.get("run_kind") == "real_data_cross_validation", "Expected real-data CV")
    require(manifest["protocol"]["strategy"] == "grouped", "Expected recorded grouped protocol")
    require(digest(data_path) == manifest["input_sha256"], "Raw input checksum differs")
    for name, expected in manifest["artifacts"].items():
        verified_file(run_dir, name, expected)
    require(digest(run_dir / "folds.csv") == manifest["split_sha256"], "Split checksum differs")
    expected_pairs = {
        (seed, fold)
        for seed in manifest["protocol"]["model_seeds"]
        for fold in range(manifest["protocol"]["folds"])
    }
    require(
        {(item["seed"], item["fold"]) for item in manifest["fold_runs"]} == expected_pairs
        and len(manifest["fold_runs"]) == len(expected_pairs),
        "Fold-run identities differ",
    )
    for item in manifest["fold_runs"]:
        name = item["directory"]
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts, "Unsafe fold-run directory"
        )
        child_dir = run_dir / relative
        require(child_dir.resolve().is_relative_to(run_dir.resolve()), "Fold directory escapes run")
        require(
            not any(
                (run_dir.joinpath(*relative.parts[:i])).is_symlink()
                for i in range(1, len(relative.parts) + 1)
            ),
            "Symlink fold directory",
        )
        child_path = verified_file(child_dir, "manifest.json", item["manifest_sha256"])
        child = json.loads(child_path.read_text())
        require(child["input_sha256"] == manifest["input_sha256"], "Child input differs")
        require(
            child["artifacts"]["model.pt"] == item["checkpoint_sha256"],
            "Checkpoint identity differs",
        )
        verified_file(child_dir, "model.pt", item["checkpoint_sha256"])
    metadata = read_metadata(data_path)
    require(len(metadata) == manifest["rows"], "Recorded row population differs")
    values = validate_population(
        read_csv(run_dir / "ensemble-oof.csv"),
        read_csv(run_dir / "oof.csv"),
        read_csv(run_dir / "folds.csv"),
        metadata,
        manifest["protocol"],
    )
    result = summarize(*values)
    retained = json.loads((run_dir / "analysis.json").read_text())["pooled_oof"]["log_loss"]
    require(
        np.isclose(result["pooled_log_loss"], retained, rtol=0, atol=1e-12),
        "Pooled log loss differs",
    )
    return {
        "schema_version": 1,
        "project_id": "iceberg",
        "analysis_kind": "retained_oof_hard_cases",
        "producer_manifest_sha256": digest(manifest_path),
        "producer_source": manifest["source"],
        "input_sha256": manifest["input_sha256"],
        "split_sha256": manifest["split_sha256"],
        "oof_sha256": manifest["artifacts"]["ensemble-oof.csv"],
        "analyzer_sha256": digest(Path(__file__)),
        "prediction_reconciliation": "two-seed mean verified",
        "results": result,
        "limitations": [
            "Error-selected diagnostic cohort, not new test performance or an improvement.",
            "Exact angle is an acquisition proxy, not a verified scene ID.",
            "Two seeds share one fold plan; overlap is descriptive, not a confidence estimate.",
            "No data filtering, relabelling, retraining or model promotion performed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(analyze(args.run_dir, args.data), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
