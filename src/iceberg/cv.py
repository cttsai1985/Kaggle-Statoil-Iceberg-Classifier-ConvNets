"""Fixed-recipe cross-validation, acquisition-group auditing, and OOF analysis."""

import contextlib
import csv
import json
import multiprocessing
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from iceberg import PROJECT_ID
from iceberg.data import Preprocessing, RadarDataset, load_data, sha256
from iceberg.runtime import (
    angle_overlap,
    load_model,
    probabilities,
    resolve_config,
    select_device,
    source_identity,
    train,
    validate_split_indices,
    verify_run,
    write_csv,
    write_json,
)


def acquisition_groups(data) -> np.ndarray:
    """Exact observed angles proxy acquisitions; every missing angle is its own group."""
    return np.asarray(
        [
            f"missing:{identifier}" if np.isnan(angle) else f"angle:{float(angle).hex()}"
            for identifier, angle in zip(data.ids, data.angles[:, 0], strict=True)
        ]
    )


def build_folds(data, *, folds=5, strategy="stratified", split_seed=25):
    if type(folds) is not int or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if strategy not in {"stratified", "grouped"}:
        raise ValueError("strategy must be stratified or grouped")
    if min(np.bincount(data.targets.astype(int), minlength=2)) < folds:
        raise ValueError("Each class needs at least one record per fold")
    groups = acquisition_groups(data)
    if strategy == "grouped":
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=split_seed)
        splits = splitter.split(np.zeros(len(data.ids)), data.targets, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
        splits = splitter.split(np.zeros(len(data.ids)), data.targets)
    result = []
    coverage = np.zeros(len(data.ids), dtype=int)
    for fit, validation in splits:
        fit, validation = validate_split_indices(len(data.ids), (fit, validation))
        if len(np.unique(data.targets[validation])) != 2:
            raise ValueError("A fold has only one class; use fewer folds for this grouped dataset")
        if strategy == "grouped" and np.intersect1d(groups[fit], groups[validation]).size:
            raise ValueError("Acquisition groups overlap")
        coverage[validation] += 1
        result.append((fit, validation))
    if not np.all(coverage == 1):
        raise ValueError("Every row must receive exactly one validation assignment")
    return result


def split_audit(data, splits):
    return [
        {
            "fold": fold,
            "train_rows": len(fit),
            "validation_rows": len(validation),
            "train_positives": int(data.targets[fit].sum()),
            "validation_positives": int(data.targets[validation].sum()),
            "validation_positive_fraction": float(data.targets[validation].mean()),
            **angle_overlap(data, fit, validation),
        }
        for fold, (fit, validation) in enumerate(splits)
    ]


def classification_metrics(targets, probabilities):
    return {
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(targets, probabilities)),
        "roc_auc": float(roc_auc_score(targets, probabilities))
        if len(np.unique(targets)) == 2
        else None,
    }


def summarize_oof(rows):
    """Report fixed probability bins and row-level errors; do not fit calibration on OOF."""
    targets = np.asarray([int(row["target"]) for row in rows])
    probabilities = np.asarray([float(row["probability"]) for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    missing = np.asarray([int(row["angle_missing"]) for row in rows], dtype=bool)
    if not len(rows) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("OOF records must be nonempty and unique")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("OOF probabilities are invalid")
    fold_metrics = [
        {
            "fold": int(fold),
            "rows": int((folds == fold).sum()),
            **classification_metrics(targets[folds == fold], probabilities[folds == fold]),
        }
        for fold in np.unique(folds)
    ]
    variability = {}
    for metric in ["log_loss", "brier_score", "roc_auc"]:
        values = [item[metric] for item in fold_metrics if item[metric] is not None]
        variability[metric] = (
            {
                "mean": float(np.mean(values)),
                "std_ddof1": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "min": float(min(values)),
                "max": float(max(values)),
            }
            if values
            else None
        )
    bins, ece = [], 0.0
    assignments = np.minimum((probabilities * 10).astype(int), 9)
    for number in range(10):
        mask = assignments == number
        count = int(mask.sum())
        predicted = float(probabilities[mask].mean()) if count else None
        observed = float(targets[mask].mean()) if count else None
        if count:
            ece += count / len(rows) * abs(predicted - observed)
        bins.append(
            {
                "lower": number / 10,
                "upper": (number + 1) / 10,
                "rows": count,
                "mean_probability": predicted,
                "observed_positive_fraction": observed,
            }
        )
    strata = {}
    for name, mask in [
        ("missing_angle", missing),
        ("known_angle", ~missing),
        ("ship", targets == 0),
        ("iceberg", targets == 1),
    ]:
        if mask.any():
            strata[name] = {
                "rows": int(mask.sum()),
                "positives": int(targets[mask].sum()),
                **classification_metrics(targets[mask], probabilities[mask]),
            }
    clipped = probabilities.clip(np.finfo(float).eps, 1 - np.finfo(float).eps)
    losses = -(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped))
    errors = [
        {
            **rows[int(i)],
            "row_log_loss": float(losses[i]),
            "absolute_error": float(abs(targets[i] - probabilities[i])),
        }
        for i in np.argsort(-losses, kind="stable")
    ]
    return {
        "rows": len(rows),
        "positives": int(targets.sum()),
        "negatives": int((targets == 0).sum()),
        "positive_fraction": float(targets.mean()),
        "missing_angle_rows": int(missing.sum()),
        "pooled_oof": classification_metrics(targets, probabilities),
        "fold_metrics": fold_metrics,
        "fold_distribution": variability,
        "variability_interpretation": (
            "Descriptive fold dispersion, not a confidence interval; training sets overlap."
        ),
        "calibration_bins": bins,
        "expected_calibration_error_10_bins": float(ece),
        "calibration_interpretation": (
            "Diagnostic only; no calibrator was fitted to these OOF targets."
        ),
        "strata": strata,
    }, errors


def _fit_fold(data_path, run_dir, config, fit, validation, fixture, metadata):
    with (run_dir.parent / (run_dir.name + ".log")).open("w") as log:
        with contextlib.redirect_stdout(log):
            return train(
                data_path,
                run_dir,
                config,
                fixture=fixture,
                split_indices=(fit, validation),
                checkpoint_selection="last",
                split_metadata=metadata,
            )


OOF_COLUMNS = ["id", "fold", "target", "probability", "inc_angle", "angle_missing", "group"]
CV_ARTIFACTS = {
    "cv-config.json",
    "folds.csv",
    "grouped-audit-splits.csv",
    "oof.csv",
    "ensemble-oof.csv",
    "analysis.json",
    "errors.csv",
}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def analyze_predictions(rows, seeds):
    """Pair predictions by record, preserving identical folds across predeclared seeds."""
    if {int(row["seed"]) for row in rows} != set(seeds):
        raise ValueError("OOF model seeds differ from the protocol")
    seed_rows = [
        [{key: row[key] for key in OOF_COLUMNS} for row in rows if int(row["seed"]) == seed]
        for seed in seeds
    ]
    seed_results = []
    for seed, records in zip(seeds, seed_rows, strict=True):
        summary, _ = summarize_oof(records)
        seed_results.append({"seed": seed, **summary})
    reference = seed_rows[0]
    metadata = [key for key in OOF_COLUMNS if key != "probability"]
    for records in seed_rows[1:]:
        if len(records) != len(reference) or any(
            any(left[key] != right[key] for key in metadata)
            for left, right in zip(reference, records, strict=True)
        ):
            raise ValueError("Seeds do not share the same OOF rows, targets, and fold assignments")
    ensemble = [
        {
            **row,
            "probability": str(
                float(np.mean([float(records[index]["probability"]) for records in seed_rows]))
            ),
        }
        for index, row in enumerate(reference)
    ]
    analysis, errors = summarize_oof(ensemble)
    analysis["prediction_kind"] = "mean_probability_across_predeclared_seeds"
    analysis["seed_results"] = seed_results
    analysis["seed_distribution"] = {}
    for metric in ["log_loss", "brier_score", "roc_auc"]:
        values = [record["pooled_oof"][metric] for record in seed_results]
        analysis["seed_distribution"][metric] = {
            "mean": float(np.mean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "range": float(max(values) - min(values)),
            "std_ddof1": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    analysis["seed_interpretation"] = (
        "Descriptive training-seed variation on identical folds, not independent replicates "
        "or a confidence interval. All seeds are retained; no fold or seed is selected."
    )
    return analysis, errors, ensemble


def cross_validate(
    data_path: Path,
    run_dir: Path,
    config: dict,
    *,
    folds=5,
    strategy="stratified",
    split_seed=25,
    seeds=(1017, 2027),
    workers=1,
    fixture=False,
):
    """Evaluate fixed final-epoch models; outer-fold labels never select checkpoints."""
    config = resolve_config(config)
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Model seeds must be a nonempty sequence of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Model seeds must be unique")
    if config["max_train_rows"] is not None:
        raise ValueError("CV requires all training rows; max_train_rows must be null")
    if type(workers) is not int or not 1 <= workers <= folds * len(seeds):
        raise ValueError("workers must be between one and the number of fits")
    if workers > 1 and select_device(config["device"]).type == "cuda":
        raise ValueError("Use workers=1 for a single GPU, or explicitly select CPU workers")
    data = load_data(data_path, labeled=True)
    splits = build_folds(data, folds=folds, strategy=strategy, split_seed=split_seed)
    grouped = (
        splits
        if strategy == "grouped"
        else build_folds(data, folds=folds, strategy="grouped", split_seed=split_seed)
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assignments = np.full(len(data.ids), -1, dtype=int)
    grouped_assignments = np.full(len(data.ids), -1, dtype=int)
    for fold, (_, validation) in enumerate(splits):
        assignments[validation] = fold
    for fold, (_, validation) in enumerate(grouped):
        grouped_assignments[validation] = fold
    groups = acquisition_groups(data)
    write_csv(
        run_dir / "folds.csv",
        ["id", "fold", "group"],
        zip(data.ids, assignments, groups, strict=True),
    )
    write_csv(
        run_dir / "grouped-audit-splits.csv",
        ["id", "fold", "group"],
        zip(data.ids, grouped_assignments, groups, strict=True),
    )
    protocol = {
        "folds": folds,
        "strategy": strategy,
        "shuffle": True,
        "split_seed": split_seed,
        "model_seeds": seeds,
        "workers": workers,
        "threads_per_worker": config["threads"],
        "checkpoint_selection": "last",
        "config": config,
    }
    write_json(run_dir / "cv-config.json", protocol)
    manifest = {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "logical_run_id": str(uuid.uuid4()),
        "status": "running",
        "run_kind": "synthetic_cv_fixture" if fixture else "real_data_cross_validation",
        "source": source_identity(),
        "input_sha256": sha256(data_path),
        "rows": len(data.ids),
        "protocol": protocol,
        "protocol_sha256": sha256(run_dir / "cv-config.json"),
        "split_sha256": sha256(run_dir / "folds.csv"),
        "split_audit": split_audit(data, splits),
        "grouped_split_audit": split_audit(data, grouped),
        "grouped_model_evaluation": "included" if strategy == "grouped" else "not_run",
        "grouping_limitations": (
            "Exact incidence angle is an acquisition proxy, not a verified scene identifier; "
            "missing angles are individual groups."
        ),
        "interpretation": (
            "Fixed-recipe OOF evaluation; model choices informed by earlier runs on these data, "
            "not an untouched external test."
        ),
        "fold_runs": [],
    }
    write_json(run_dir / "cv-manifest.json", manifest)
    try:
        jobs = []
        for seed in seeds:
            (run_dir / f"seed-{seed}").mkdir()
            for fold, (fit, validation) in enumerate(splits):
                metadata = {
                    "strategy": strategy,
                    "fold": fold,
                    "folds": folds,
                    "split_seed": split_seed,
                    "parent_cv_run_id": manifest["logical_run_id"],
                }
                child = run_dir / f"seed-{seed}" / f"fold-{fold:02d}"
                jobs.append(
                    (data_path, child, {**config, "seed": seed}, fit, validation, fixture, metadata)
                )
        if workers == 1:
            for job in jobs:
                _fit_fold(*job)
                print(f"Completed {job[1].relative_to(run_dir)}", flush=True)
        else:
            thread_variables = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
            previous = {name: os.environ.get(name) for name in thread_variables}
            try:
                for name in thread_variables:
                    os.environ[name] = str(config["threads"])
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                ) as executor:
                    pending = {
                        executor.submit(_fit_fold, *job): job[1].relative_to(run_dir)
                        for job in jobs
                    }
                    for future in as_completed(pending):
                        future.result()
                        print(f"Completed {pending[future]}", flush=True)
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        oof = []
        for seed in seeds:
            predictions = np.full(len(data.ids), np.nan)
            for fold, (_, validation) in enumerate(splits):
                child = run_dir / f"seed-{seed}" / f"fold-{fold:02d}"
                child_manifest = verify_run(child, data_path)
                rows = read_csv(child / "validation.csv")
                if [row["id"] for row in rows] != [data.ids[i] for i in validation]:
                    raise ValueError("Fold predictions do not match the assigned validation IDs")
                predictions[validation] = [float(row["probability"]) for row in rows]
                manifest["fold_runs"].append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "directory": str(child.relative_to(run_dir)),
                        "manifest_sha256": sha256(child / "manifest.json"),
                        "checkpoint_sha256": child_manifest["artifacts"]["model.pt"],
                        "logical_run_id": child_manifest["logical_run_id"],
                    }
                )
            oof.extend(
                (
                    seed,
                    identifier,
                    int(assignments[i]),
                    int(data.targets[i]),
                    float(predictions[i]),
                    "" if np.isnan(data.angles[i, 0]) else float(data.angles[i, 0]),
                    int(np.isnan(data.angles[i, 0])),
                    groups[i],
                )
                for i, identifier in enumerate(data.ids)
            )
        write_csv(run_dir / "oof.csv", ["seed", *OOF_COLUMNS], oof)
        analysis, errors, ensemble = analyze_predictions(read_csv(run_dir / "oof.csv"), seeds)
        analysis.update(
            split_audit=manifest["split_audit"],
            grouped_split_audit=manifest["grouped_split_audit"],
            grouped_model_evaluation=manifest["grouped_model_evaluation"],
        )
        write_csv(
            run_dir / "ensemble-oof.csv",
            OOF_COLUMNS,
            ([row[column] for column in OOF_COLUMNS] for row in ensemble),
        )
        write_json(run_dir / "analysis.json", analysis)
        error_columns = [*OOF_COLUMNS, "row_log_loss", "absolute_error"]
        write_csv(
            run_dir / "errors.csv",
            error_columns,
            ([row[column] for column in error_columns] for row in errors),
        )
        manifest.update(
            status="completed",
            elapsed_seconds=time.monotonic() - started,
            artifacts={name: sha256(run_dir / name) for name in sorted(CV_ARTIFACTS)},
        )
        write_json(run_dir / "cv-manifest.json", manifest)
        analyze_cv(run_dir, data_path)
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(run_dir / "cv-manifest.json", manifest)
        raise


def analyze_cv(run_dir: Path, data_path: Path | None = None) -> dict:
    """Verify the parent/child artifact chain and recompute analysis from retained OOF rows."""
    manifest = json.loads((run_dir / "cv-manifest.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("schema_version") != 1:
        raise ValueError("Expected completed CV evidence")
    if set(manifest["artifacts"]) != CV_ARTIFACTS:
        raise ValueError("Unexpected CV artifact manifest")
    for name, digest in manifest["artifacts"].items():
        if (run_dir / name).is_symlink() or sha256(run_dir / name) != digest:
            raise ValueError(f"CV artifact checksum mismatch: {name}")
    protocol = manifest["protocol"]
    if (
        protocol != json.loads((run_dir / "cv-config.json").read_text())
        or manifest["protocol_sha256"] != manifest["artifacts"]["cv-config.json"]
        or manifest["split_sha256"] != manifest["artifacts"]["folds.csv"]
    ):
        raise ValueError("CV protocol or split fingerprint differs")
    if data_path is not None and sha256(data_path) != manifest["input_sha256"]:
        raise ValueError("CV input differs")
    seeds = protocol["model_seeds"]
    expected_fits = [(seed, fold) for seed in seeds for fold in range(protocol["folds"])]
    if len(manifest["fold_runs"]) != len(expected_fits):
        raise ValueError("Incomplete fold evidence")
    assignments = read_csv(run_dir / "folds.csv")
    if (
        len(assignments) != manifest["rows"]
        or len({row["id"] for row in assignments}) != manifest["rows"]
    ):
        raise ValueError("Fold assignments do not cover each record exactly once")
    rows = read_csv(run_dir / "oof.csv")
    if len(rows) != manifest["rows"] * len(seeds):
        raise ValueError("OOF coverage differs")
    for (seed, fold), record in zip(expected_fits, manifest["fold_runs"], strict=True):
        if (record["seed"], record["fold"], record["directory"]) != (
            seed,
            fold,
            f"seed-{seed}/fold-{fold:02d}",
        ):
            raise ValueError("Invalid fold artifact link")
        child = run_dir / record["directory"]
        if (
            child.is_symlink()
            or child.parent.is_symlink()
            or sha256(child / "manifest.json") != record["manifest_sha256"]
        ):
            raise ValueError("Child manifest checksum mismatch")
        verified = verify_run(child, data_path)
        if (
            verified["input_sha256"] != manifest["input_sha256"]
            or verified["source"]["python_source_sha256"]
            != manifest["source"]["python_source_sha256"]
        ):
            raise ValueError("Child input or source fingerprint differs")
        if (
            verified["checkpoint_selection"] != "last"
            or verified["config"] != {**protocol["config"], "seed": seed}
            or verified["artifacts"]["model.pt"] != record["checkpoint_sha256"]
        ):
            raise ValueError("Fold recipe or checkpoint differs from the fixed protocol")
        predictions = read_csv(child / "validation.csv")
        expected_ids = [row["id"] for row in assignments if int(row["fold"]) == fold]
        selected = [row for row in rows if int(row["seed"]) == seed and int(row["fold"]) == fold]
        if [row["id"] for row in predictions] != expected_ids or [
            row["id"] for row in selected
        ] != expected_ids:
            raise ValueError("OOF predictions differ from fold assignments")
        for prediction, row in zip(predictions, selected, strict=True):
            if any(prediction[key] != row[key] for key in ["id", "target", "probability"]):
                raise ValueError("OOF prediction differs from child evidence")
    analysis, _, ensemble = analyze_predictions(rows, seeds)
    if ensemble != read_csv(run_dir / "ensemble-oof.csv"):
        raise ValueError("Stored ensemble differs from paired seed predictions")
    analysis.update(
        split_audit=manifest["split_audit"],
        grouped_split_audit=manifest["grouped_split_audit"],
        grouped_model_evaluation=manifest["grouped_model_evaluation"],
    )
    if analysis != json.loads((run_dir / "analysis.json").read_text()):
        raise ValueError("Stored analysis differs from retained OOF predictions")
    return analysis


def predict_cv(run_dir: Path, data_path: Path, output: Path, *, device: str = "auto") -> dict:
    """Average every verified fold/seed model using its own fitted preprocessing."""
    analyze_cv(run_dir)
    manifest_path = run_dir / "cv-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_digest = sha256(manifest_path)
    data = load_data(data_path, labeled=False)
    training_ids = {row["id"] for row in read_csv(run_dir / "folds.csv")}
    if training_ids.intersection(data.ids):
        raise ValueError(
            "Inference input IDs overlap the CV training data; use retained OOF predictions"
        )
    protected = {data_path.resolve(), manifest_path.resolve()}
    protected.update((run_dir / name).resolve() for name in manifest["artifacts"])
    for record in manifest["fold_runs"]:
        child = run_dir / record["directory"]
        protected.update(path.resolve() for path in child.iterdir() if path.is_file())
    if output.resolve() in protected or output.with_suffix(".receipt.json").resolve() in protected:
        raise ValueError("Prediction output would overwrite input or retained CV evidence")
    combined = np.zeros(len(data.ids), dtype=np.float64)
    checkpoints = []
    tta = manifest["protocol"]["config"]["tta"]
    for record in manifest["fold_runs"]:
        checkpoint = run_dir / record["directory"] / "model.pt"
        model, saved = load_model(checkpoint, device=device)
        prediction = probabilities(
            model,
            RadarDataset(data, range(len(data.ids)), Preprocessing(**saved["preprocessing"])),
            saved["config"]["batch_size"],
            tta=tta,
        )
        if (
            prediction.shape != combined.shape
            or not np.isfinite(prediction).all()
            or np.any((prediction < 0) | (prediction > 1))
        ):
            raise ValueError("A fold produced invalid or incomplete inference probabilities")
        combined += prediction
        checkpoints.append(
            {
                "seed": record["seed"],
                "fold": record["fold"],
                "directory": record["directory"],
                "sha256": record["checkpoint_sha256"],
            }
        )
        del model, saved
    combined /= len(checkpoints)
    write_csv(output, ["id", "is_iceberg"], zip(data.ids, combined, strict=True))
    receipt = {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "status": "cv_ensemble_predictions",
        "cv_logical_run_id": manifest["logical_run_id"],
        "cv_manifest_sha256": manifest_digest,
        "training_input_sha256": manifest["input_sha256"],
        "input_sha256": sha256(data_path),
        "output_sha256": sha256(output),
        "rows": len(data.ids),
        "model_count": len(checkpoints),
        "aggregation": "equal_weight_mean_probability_across_all_folds_and_seeds",
        "tta": tta,
        "device": str(select_device(device)),
        "checkpoints": checkpoints,
        "training_source": manifest["source"],
        "inference_source": source_identity(),
    }
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt
