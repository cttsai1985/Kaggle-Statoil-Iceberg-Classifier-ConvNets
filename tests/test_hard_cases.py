"""Small known-error examples verify diagnostic arithmetic and provenance rejection."""

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "iceberg_hard_cases", Path(__file__).resolve().parents[1] / "scripts/analyze_hard_cases.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_known_log_loss_concentration_and_seed_overlap():
    values = analysis.summarize(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.6, 0.9],
        [0, 0, 1, 1],
        [34, 42, np.nan, 37],
        {1: [0.1, 0.8, 0.6, 0.9], 2: [0.15, 0.75, 0.65, 0.85]},
    )
    expected_sum = -2 * math.log(0.9) - math.log(0.2) - math.log(0.6)
    assert values["pooled_log_loss"] == pytest.approx(expected_sum / 4)
    assert values["hard_cohort"]["rows"] == 1
    assert values["hard_cohort"]["ship_rows"] == 1
    assert values["hard_cohort"]["share_of_total_loss"] == pytest.approx(
        -math.log(0.2) / expected_sum
    )
    assert values["seed_error_overlap"]["jaccard"] == 1.0
    assert values["strata"]["missing_angle"]["hard_rows"] == 0
    assert np.array_equal(analysis.hardest(np.ones(4)), [True, False, False, False])


def population():
    metadata = {"a": (0, 34.0), "b": (1, np.nan)}
    rows, assignments, seeds = [], [], []
    for identifier, (target, angle) in metadata.items():
        fold = 0 if identifier == "a" else 1
        group = f"angle:{angle.hex()}" if np.isfinite(angle) else f"missing:{identifier}"
        row = {
            "id": identifier,
            "target": str(target),
            "fold": str(fold),
            "probability": "0.3" if target == 0 else "0.7",
            "inc_angle": "34.0" if target == 0 else "",
            "angle_missing": str(int(np.isnan(angle))),
            "group": group,
        }
        rows.append(row)
        assignments.append({key: row[key] for key in ["id", "fold", "group"]})
        seeds.extend([{**row, "seed": str(seed)} for seed in [1, 2]])
    return rows, seeds, assignments, metadata, {"model_seeds": [1, 2], "folds": 2}


def test_population_prediction_and_label_checks_fail_closed():
    values = population()
    analysis.validate_population(*values)
    changed = copy.deepcopy(values)
    changed[0][0]["target"] = "1"
    with pytest.raises(ValueError, match="label differs"):
        analysis.validate_population(*changed)
    changed = copy.deepcopy(values)
    changed[1][0]["probability"] = "0.9"
    with pytest.raises(ValueError, match="seed mean"):
        analysis.validate_population(*changed)
    changed = copy.deepcopy(values)
    changed[1].append(changed[1][0])
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.validate_population(*changed)


def test_invalid_probabilities_and_tampered_artifacts_rejected(tmp_path):
    for prediction in [[float("nan")], [1.1]]:
        with pytest.raises(ValueError, match="probabilities"):
            analysis.row_losses([0], prediction)
    with pytest.raises(ValueError, match="binary"):
        analysis.row_losses([2], [0.5])
    path = tmp_path / "oof.csv"
    path.write_text("original")
    expected = analysis.digest(path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        analysis.verified_file(tmp_path, "oof.csv", expected)
    with pytest.raises(ValueError, match="basename"):
        analysis.verified_file(tmp_path, "../escape", expected)
