"""One bounded zoom ablation; reuse the maintained grouped-CV training functions."""

import argparse
import json
from pathlib import Path

import numpy as np

from iceberg.cv import analyze_cv, classification_metrics, cross_validate, read_csv
from iceberg.data import keras_zoom, load_data, sha256
from iceberg.runtime import evaluate, resolve_config, source_identity, write_json

POLICIES = ("none", "mild_5pct", "extreme_5pct")
SEEDS = (1017, 2027)


def probe():
    y, x = np.indices((75, 75), dtype=np.float32)
    image = np.stack([x, y])
    extreme = keras_zoom(image, np.array([0.1, 0.1]))
    identity = keras_zoom(image, np.ones(2))
    return {
        "fixture": "synthetic 75x75 coordinate grid; no competition pixels",
        "identity_exact": bool(np.array_equal(identity, image)),
        "extreme_sampled_x_range": [float(extreme[0].min()), float(extreme[0].max())],
        "extreme_sampled_y_range": [float(extreme[1].min()), float(extreme[1].max())],
        "legacy_transform": "Keras 2 centered inverse map; order=0 and nearest fill",
    }


def metrics(targets, prediction, known):
    return {
        "all_rows": classification_metrics(targets, prediction),
        "known_angle": classification_metrics(targets[known], prediction[known]),
        "missing_angle": classification_metrics(targets[~known], prediction[~known]),
    }


def analyze(data_path: Path, output: Path):
    data = load_data(data_path, labeled=True)
    known = np.isfinite(data.angles[:, 0])
    targets = data.targets.astype(int)
    plan = json.loads((output / "plan.json").read_text())
    if plan["input_sha256"] != sha256(data_path):
        raise ValueError("Study input differs from its frozen plan")
    conditions, predictions, reload_checks = {}, {}, []
    for policy in POLICIES:
        directory = output / policy
        analyze_cv(directory, data_path)
        manifest = json.loads((directory / "cv-manifest.json").read_text())
        if manifest["source"] != plan["source"]:
            raise ValueError("Training source differs from the frozen study source")
        expected = plan["config"] | {"zoom_policy": policy, "name": f"zoom-study-{policy}"}
        if manifest["protocol"]["config"] != resolve_config(expected):
            raise ValueError("A condition changed more than its declared zoom policy")
        if manifest["split_sha256"] != plan["split_sha256"]:
            raise ValueError("Conditions did not use identical folds")
        rows = read_csv(directory / "oof.csv")
        ensemble = read_csv(directory / "ensemble-oof.csv")
        if [row["id"] for row in ensemble] != data.ids:
            raise ValueError("OOF ensemble does not cover every input in order")
        probability = np.array([float(row["probability"]) for row in ensemble])
        predictions[policy] = probability
        seed_metrics, fold_metrics = {}, []
        for seed in SEEDS:
            seed_rows = [row for row in rows if int(row["seed"]) == seed]
            if [row["id"] for row in seed_rows] != data.ids:
                raise ValueError("Seed OOF coverage mismatch")
            seed_p = np.array([float(row["probability"]) for row in seed_rows])
            seed_metrics[str(seed)] = metrics(targets, seed_p, known)
            for fold in range(5):
                mask = np.array([int(row["fold"]) == fold for row in seed_rows])
                fold_metrics.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        **metrics(targets[mask], seed_p[mask], known[mask]),
                    }
                )
        for child in manifest["fold_runs"]:
            run = directory / child["directory"]
            saved = read_csv(run / "validation.csv")
            reloaded = run / "independent-reload.csv"
            result = evaluate(run / "model.pt", data_path, reloaded, device="cpu")
            rebuilt = read_csv(reloaded)
            if [row["id"] for row in saved] != [row["id"] for row in rebuilt]:
                raise ValueError("Reloaded validation membership changed")
            delta = max(
                abs(float(a["probability"]) - float(b["probability"]))
                for a, b in zip(saved, rebuilt, strict=True)
            )
            if delta > 1e-7:
                raise ValueError("Independent checkpoint reconstruction differs")
            reload_checks.append(
                {
                    "policy": policy,
                    "seed": child["seed"],
                    "fold": child["fold"],
                    "max_probability_delta": delta,
                    "log_loss": result["validation_log_loss"],
                    "checkpoint_sha256": sha256(run / "model.pt"),
                }
            )
        clipped = np.clip(probability, np.finfo(float).eps, 1 - np.finfo(float).eps)
        loss = -(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped))
        hard_n = int(np.ceil(0.1 * len(loss)))
        conditions[policy] = {
            "ensemble": metrics(targets, probability, known),
            "seeds": seed_metrics,
            "folds": fold_metrics,
            "hardest_10pct_loss_share": float(np.sort(loss)[-hard_n:].sum() / loss.sum()),
            "oof_sha256": sha256(directory / "oof.csv"),
            "cv_manifest_sha256": sha256(directory / "cv-manifest.json"),
            "elapsed_seconds": manifest["elapsed_seconds"],
        }
    comparisons = {}
    for before, after in [
        ("none", "mild_5pct"),
        ("none", "extreme_5pct"),
        ("mild_5pct", "extreme_5pct"),
    ]:
        left, right = conditions[before], conditions[after]
        comparisons[f"{after}_minus_{before}"] = {
            "ensemble": {
                population: right["ensemble"][population]["log_loss"]
                - left["ensemble"][population]["log_loss"]
                for population in ["all_rows", "known_angle", "missing_angle"]
            },
            "known_angle_seed_deltas": {
                str(seed): right["seeds"][str(seed)]["known_angle"]["log_loss"]
                - left["seeds"][str(seed)]["known_angle"]["log_loss"]
                for seed in SEEDS
            },
            "known_angle_paired_fold_deltas": [
                {
                    "seed": a["seed"],
                    "fold": a["fold"],
                    "delta": b["known_angle"]["log_loss"] - a["known_angle"]["log_loss"],
                }
                for a, b in zip(left["folds"], right["folds"], strict=True)
            ],
        }
    receipt = {
        "schema_version": 1,
        "study": "zoom severity at matched 5% trigger probability",
        "plan": plan,
        "plan_sha256": sha256(output / "plan.json"),
        "analyzer_sha256": sha256(Path(__file__)),
        "probe": probe(),
        "rows": len(targets),
        "known_angle_rows": int(known.sum()),
        "missing_angle_rows": int((~known).sum()),
        "conditions": conditions,
        "comparisons": comparisons,
        "independent_reload_checks": reload_checks,
        "limits": [
            "Retrospective modern-model study, not a replay of Keras weights/training.",
            "No leaderboard submission; OOF deltas do not establish a leaderboard effect.",
            "Shared folds and two seeds: fold dispersion is not a confidence interval.",
            "The normal pipeline is unchanged. All study arms remove its coupled mild scale, "
            "then test a separate Keras-style zoom at the same 5% trigger probability.",
            "All arms consume the same random draws; none uses identity zoom at those triggers.",
            "Historical author intent cannot be inferred from the validation result.",
        ],
    }
    write_json(output / "summary.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        config = json.loads(Path("configs/metadata.json").read_text()) | {
            "device": "cpu",
            "threads": 2,
            "deterministic": True,
            "mixed_precision": False,
        }
        source = source_identity()
        if source["git_dirty"]:
            raise ValueError("Commit the predeclared source/protocol before fitting")
        args.output.mkdir(parents=True, exist_ok=False)
        # Freeze the exact assignment hash without fitting a model.
        from iceberg.cv import acquisition_groups, build_folds
        from iceberg.runtime import write_csv

        data = load_data(args.data, labeled=True)
        splits = build_folds(data, folds=5, strategy="grouped", split_seed=25)
        folds = np.full(len(data.ids), -1, dtype=int)
        for fold, (_, validation) in enumerate(splits):
            folds[validation] = fold
        groups = acquisition_groups(data)
        write_csv(
            args.output / "folds.csv",
            ["id", "fold", "group"],
            zip(data.ids, folds, groups, strict=True),
        )
        plan = {
            "source": source,
            "script_sha256": sha256(Path(__file__)),
            "input_sha256": sha256(args.data),
            "split_sha256": sha256(args.output / "folds.csv"),
            "config": config,
            "policies": list(POLICIES),
            "seeds": list(SEEDS),
            "folds": 5,
            "split_seed": 25,
            "strategy": "grouped",
            "workers": 4,
            "maximum_training_threads": 8,
            "epochs_per_model": 30,
            "models": 30,
            "checkpoint_selection": "last",
            "primary": "known_angle log_loss",
            "trigger_probability": 0.05,
        }
        write_json(args.output / "plan.json", plan)
        for policy in POLICIES:
            print(f"Starting {policy}: 10 fits, fixed 30 epochs", flush=True)
            cross_validate(
                args.data,
                args.output / policy,
                config | {"zoom_policy": policy, "name": f"zoom-study-{policy}"},
                folds=5,
                strategy="grouped",
                split_seed=25,
                seeds=SEEDS,
                workers=4,
            )
    result = analyze(args.data, args.output)
    print(json.dumps({key: value["ensemble"] for key, value in result["conditions"].items()}))


if __name__ == "__main__":
    main()
