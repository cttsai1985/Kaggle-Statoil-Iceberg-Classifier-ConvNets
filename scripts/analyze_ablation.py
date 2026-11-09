"""Compare matched completed CV runs; change inference TTA without refitting models."""

import argparse
import json
from pathlib import Path

import torch

from iceberg.cv import analyze_cv, analyze_predictions, classification_metrics, read_csv
from iceberg.data import Preprocessing, RadarDataset, load_data, sha256
from iceberg.runtime import environment, load_model, probabilities, write_csv, write_json


def compare(metadata_run: Path, image_run: Path, data_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    runs = {"image_only": image_run, "image_and_angle": metadata_run}
    manifests = {}
    studies = {}
    prediction_rows = {}
    parameter_counts = {}
    torch.set_num_threads(4)
    data = load_data(data_path, labeled=True)
    for name, directory in runs.items():
        analysis = analyze_cv(directory, data_path)
        manifest = json.loads((directory / "cv-manifest.json").read_text())
        manifests[name] = manifest
        if manifest["input_sha256"] != sha256(data_path):
            raise ValueError("Training input identity differs")
        studies[name + "_flips"] = analysis
        prediction_rows[name + "_flips"] = read_csv(directory / "oof.csv")
    image, metadata = manifests["image_only"], manifests["image_and_angle"]
    if image["source"]["python_source_sha256"] != metadata["source"]["python_source_sha256"]:
        raise ValueError("Training implementation differs")
    for key in ["input_sha256", "split_sha256"]:
        if image[key] != metadata[key]:
            raise ValueError(f"Unmatched {key}")
    for key in ["folds", "strategy", "split_seed", "model_seeds", "checkpoint_selection"]:
        if image["protocol"][key] != metadata["protocol"][key]:
            raise ValueError(f"Unmatched protocol: {key}")
    left = image["protocol"]["config"]
    right = metadata["protocol"]["config"]
    differences = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
    if differences != {"name", "use_angle"} or left["use_angle"] or not right["use_angle"]:
        raise ValueError("Only the optional angle branch may differ between training recipes")
    # Same retained weights and folds, changing only inference augmentation.
    for name, directory in runs.items():
        rows = []
        original = {(int(r["seed"]), r["id"]): r for r in read_csv(directory / "oof.csv")}
        for fold in manifests[name]["fold_runs"]:
            model, saved = load_model(directory / fold["directory"] / "model.pt", device="cpu")
            count = sum(parameter.numel() for parameter in model.parameters())
            if parameter_counts.setdefault(name, count) != count:
                raise ValueError("Parameter counts differ within one training condition")
            indices = saved["validation_indices"]
            prediction = probabilities(
                model,
                RadarDataset(data, indices, Preprocessing(**saved["preprocessing"])),
                saved["config"]["batch_size"],
                tta="none",
            )
            for index, probability in zip(indices, prediction, strict=True):
                rows.append(
                    {
                        **original[(fold["seed"], data.ids[index])],
                        "probability": str(float(probability)),
                    }
                )
            print(f"Inferred {name}: seed {fold['seed']}, fold {fold['fold']}, no TTA", flush=True)
        # analyze_predictions requires the same deterministic record ordering for each seed.
        rows.sort(key=lambda row: (int(row["seed"]), int(row["fold"]), row["id"]))
        analysis, _, _ = analyze_predictions(rows, image["protocol"]["model_seeds"])
        studies[name + "_none"] = analysis
        prediction_rows[name + "_none"] = rows
        write_csv(
            output / f"{name}-no-tta-oof.csv",
            list(rows[0]),
            ([r[k] for k in rows[0]] for r in rows),
        )

    def cohort_metrics(rows):
        known = [row for row in rows if int(row["angle_missing"]) == 0]
        return {
            "rows": len(rows),
            "known_angle_rows": len(known),
            "all_row_log_loss": classification_metrics(
                [int(row["target"]) for row in rows],
                [float(row["probability"]) for row in rows],
            )["log_loss"],
            "known_angle_log_loss": classification_metrics(
                [int(row["target"]) for row in known],
                [float(row["probability"]) for row in known],
            )["log_loss"],
        }

    for name, rows in prediction_rows.items():
        _, _, ensemble = analyze_predictions(rows, image["protocol"]["model_seeds"])
        studies[name]["known_angle_fold_metrics"] = [
            {"fold": fold, **cohort_metrics([row for row in ensemble if int(row["fold"]) == fold])}
            for fold in range(image["protocol"]["folds"])
        ]
        studies[name]["seed_fold_metrics"] = [
            {
                "seed": seed,
                "fold": fold,
                **cohort_metrics(
                    [row for row in rows if int(row["seed"]) == seed and int(row["fold"]) == fold]
                ),
            }
            for seed in image["protocol"]["model_seeds"]
            for fold in range(image["protocol"]["folds"])
        ]

    def compact(analysis):
        return {
            key: analysis[key]
            for key in [
                "rows",
                "pooled_oof",
                "fold_metrics",
                "fold_distribution",
                "strata",
                "seed_distribution",
                "known_angle_fold_metrics",
                "seed_fold_metrics",
            ]
        } | {
            "seed_results": [
                {"seed": r["seed"], "pooled_oof": r["pooled_oof"], "strata": r["strata"]}
                for r in analysis["seed_results"]
            ]
        }

    comparisons = []
    for label, before, after in [
        (
            "Add angle branch, four-flip inference fixed",
            "image_only_flips",
            "image_and_angle_flips",
        ),
        (
            "Add four-flip inference, image-only weights fixed",
            "image_only_none",
            "image_only_flips",
        ),
        (
            "Add four-flip inference, metadata weights fixed",
            "image_and_angle_none",
            "image_and_angle_flips",
        ),
    ]:
        a, b = studies[before], studies[after]
        comparisons.append(
            {
                "hypothesis": label,
                "before": before,
                "after": after,
                "all_row_log_loss_delta": b["pooled_oof"]["log_loss"] - a["pooled_oof"]["log_loss"],
                "known_angle_log_loss_delta": b["strata"]["known_angle"]["log_loss"]
                - a["strata"]["known_angle"]["log_loss"],
                "per_seed_all_row_deltas": [
                    {
                        "seed": x["seed"],
                        "delta": y["pooled_oof"]["log_loss"] - x["pooled_oof"]["log_loss"],
                    }
                    for x, y in zip(a["seed_results"], b["seed_results"], strict=True)
                ],
                "per_seed_known_angle_deltas": [
                    {
                        "seed": x["seed"],
                        "delta": y["strata"]["known_angle"]["log_loss"]
                        - x["strata"]["known_angle"]["log_loss"],
                    }
                    for x, y in zip(a["seed_results"], b["seed_results"], strict=True)
                ],
                "per_seed_fold_known_angle_deltas": [
                    {
                        "seed": x["seed"],
                        "fold": x["fold"],
                        "known_angle_rows": x["known_angle_rows"],
                        "delta": y["known_angle_log_loss"] - x["known_angle_log_loss"],
                    }
                    for x, y in zip(a["seed_fold_metrics"], b["seed_fold_metrics"], strict=True)
                ],
                "fold_known_angle_deltas": [
                    {
                        "fold": x["fold"],
                        "known_angle_rows": x["known_angle_rows"],
                        "delta": y["known_angle_log_loss"] - x["known_angle_log_loss"],
                    }
                    for x, y in zip(
                        a["known_angle_fold_metrics"], b["known_angle_fold_metrics"], strict=True
                    )
                ],
                "fold_all_row_deltas": [
                    {"fold": x["fold"], "delta": y["log_loss"] - x["log_loss"]}
                    for x, y in zip(a["fold_metrics"], b["fold_metrics"], strict=True)
                ],
            }
        )
    result = {
        "schema_version": 1,
        "project_id": "iceberg",
        "study_kind": "matched_component_ablation",
        "hypotheses": [
            "Does the optional incidence-angle branch help on held-out angle groups?",
            "Does four-flip inference improve the same saved weights?",
        ],
        "protocol": image["protocol"],
        "input_sha256": image["input_sha256"],
        "split_sha256": image["split_sha256"],
        "source": {name: m["source"] for name, m in manifests.items()},
        "training_execution": {
            name: {key: m["protocol"][key] for key in ("workers", "threads_per_worker")}
            for name, m in manifests.items()
        },
        "parameter_counts": parameter_counts,
        "analysis_environment": environment(torch.device("cpu")),
        "producer_manifests": {
            name: sha256(directory / "cv-manifest.json") for name, directory in runs.items()
        },
        "analysis_script_sha256": sha256(Path(__file__)),
        "training_config_differences": sorted(differences),
        "variants": {name: compact(analysis) for name, analysis in studies.items()},
        "paired_comparisons": comparisons,
        "interpretation": {
            "primary_cohort": (
                "Known-angle rows, matching test metadata availability; all rows also retained."
            ),
            "variability": (
                "Two training seeds on identical folds; descriptive sensitivity, not "
                "confidence intervals. Split-seed variation is not measured here."
            ),
            "component_scope": (
                "Adding the angle branch changes architecture and parameter count; this "
                "measures the branch as implemented, not a causal estimate of angle "
                "information alone."
            ),
            "selection": (
                "All four conditions are reported. These post-development ablations are "
                "diagnostic, not an untouched final evaluation or a new leaderboard "
                "submission."
            ),
            "budget": (
                "Both trained conditions use 5 folds x 2 seeds x 30 fixed epochs, identical "
                "training data/assignments/optimizer/augmentation. Metadata CV used two "
                "concurrent workers; image-only CV used one, with four threads per fit. "
                "Inference variants reuse "
                "identical weights."
            ),
        },
    }
    write_json(output / "study.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-run", type=Path, required=True)
    parser.add_argument("--image-run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.metadata_run, args.image_run, args.data, args.output)


if __name__ == "__main__":
    main()
