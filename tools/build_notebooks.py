"""Author the educational walkthrough and the thin offline Kaggle entry point."""

import json
from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(nb.v4.new_markdown_cell(text))


def code(text):
    cells.append(nb.v4.new_code_cell(text))


markdown("""# Two radar bands and an incidence angle: rebuilding the Iceberg CNN

This walkthrough follows the original multi-input method and explains its data, split,
architecture and artifact decisions. The default is a **bounded real-data run**: 128 training
records, the original holdout, one epoch and one pass through training data.

That verifies a runnable method port. It does not reproduce the historical leaderboard score
or establish numerical equality with the 2017 Keras model.
See [the migration rationale](../docs/migration.md).

**Prerequisites:** the locked project environment and authorized competition files. Set
`ICEBERG_DATA_DIR` to their directory. Run all cells in order;
keep data and run artifacts outside Git.""")
markdown("""## 1. Resolve the environment before interpreting a result

The same package drives the notebook and CLI. We make paths and configuration visible;
changing an epoch count or training subset changes the experiment, not only its presentation.""")
code("""import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from sklearn.model_selection import StratifiedShuffleSplit

from iceberg.data import load_data
from iceberg.runtime import evaluate, predict, train

PROJECT = Path.cwd()
if not (PROJECT / "pyproject.toml").exists():
    PROJECT = PROJECT.parent
DATA_DIR = Path(os.environ.get("ICEBERG_DATA_DIR", PROJECT / "input"))
CONFIG = json.loads((PROJECT / "configs/bounded.json").read_text())
RUN_ROOT = Path(os.environ.get("ICEBERG_RUN_DIR", PROJECT / ".local/notebook-runs"))
RUN_DIR = RUN_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
assert (DATA_DIR / "train.json").is_file(), "Download data and set ICEBERG_DATA_DIR first"
plt.rcParams.update({"figure.figsize": (8, 3.5), "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
print(json.dumps(CONFIG, indent=2))""")
markdown("""## 2. Inspect the two data modalities

Each observation contains two 75×75 radar bands and one incidence angle. Keep both channels:
they measure different polarizations of the same scene. The original rescaling is `x / 100 + 0.5`.
It is a fixed transform, not a fitted normalizer.
The string `na` becomes the original `-1` angle sentinel.

Before training, inspect shapes, target balance and missing metadata. Raw competition rows
and radar pixels are not embedded in this notebook's saved outputs.""")
code("""data = load_data(DATA_DIR / "train.json", labeled=True)
print({"records": len(data.ids), "radar_shape": data.images.shape,
       "iceberg_fraction": float(data.targets.mean()),
       "missing_angles": int((data.angles[:, 0] == -1).sum())})
observed_angles = data.angles[data.angles[:, 0] != -1, 0]
fig, ax = plt.subplots()
ax.hist(observed_angles, bins=20, color="#176b75", edgecolor="white")
ax.set(title="Incidence-angle coverage is part of the model's input context",
       xlabel="Incidence angle (degrees; missing sentinel excluded)", ylabel="Training records")
fig.tight_layout()
plt.show()""")
markdown("""**Interpretation.** The histogram describes the observed angle coverage;
it does not show
which angles cause errors. A useful follow-up is to inspect holdout error by angle availability.
Replacing `-1` with learned imputation would be a new preprocessing experiment
and should be labeled.

## 3. Preserve the validation question

Use the original stratified split: seed 25 and a 15.9% holdout. The bounded profile reduces
only the training portion after that split, keeping the holdout intact. Predictions on the
training set are a diagnostic, not held-out validation.""")
code("""reference_split = StratifiedShuffleSplit(n_splits=1, test_size=0.159, random_state=25)
train_indices, validation_indices = next(reference_split.split(data.images, data.targets))
assert not set(train_indices) & set(validation_indices)
print({"reference_training_rows": len(train_indices), "holdout_rows": len(validation_indices),
       "bounded_training_limit": CONFIG["max_train_rows"]})""")
markdown("""## 4. Map the original architecture before changing it

```text
two radar channels ──┬─ pooled convolution branch ───────┐
                    └─ full-resolution conv branch ────┼─ dense blocks → iceberg probability
incidence angle ─────── normalization ───────────────────┘
```

Both branches retain the original depths, widths, ELU activations and pooling choices.
Batch normalization explicitly follows the original momentum-zero/population-variance convention.
Spatial augmentation applies the same geometry to both bands. The original fixed zoom factor
0.1 is preserved and documented even though it is unusually aggressive.

The output uses logits during training and sigmoid for probabilities. The optimizer's old
`decay` argument is mapped to learning-rate decay, not L2 weight decay.

## 5. Train a bounded recipe and retain its evidence

This call creates a fresh run directory. It saves the chosen configuration, input checksum,
split roles, training history, best checkpoint and held-out predictions. A failed run retains
its failed status. No private tracking service is required to complete the local run.""")
code("""manifest = train(DATA_DIR / "train.json", RUN_DIR, CONFIG)
metrics = json.loads((RUN_DIR / "metrics.json").read_text())
display(Markdown(
    f"**Observed held-out log loss: {metrics['validation_log_loss']:.4f}.** "
    f"Training used {manifest['train_rows']} records and evaluation used "
    f"{manifest['validation_rows']} untouched holdout records. "
    "This is bounded-run evidence; it is not a historical leaderboard comparison."
))""")
markdown("""## 6. Reload the checkpoint and inspect errors

Evaluation reconstructs the model in a separate call, verifies the training-file checksum,
and uses the checkpoint's saved holdout indices. A fresh CLI process exercises the same
path in the reproduction instructions. Histogram separation can help diagnose the predictions;
it does not replace log loss or establish calibration.""")
code("""reloaded = evaluate(RUN_DIR / "model.pt", DATA_DIR / "train.json")
assert abs(reloaded["validation_log_loss"] - metrics["validation_log_loss"]) < 1e-12
with (RUN_DIR / "validation.csv").open() as stream:
    validation_rows = list(csv.DictReader(stream))
fig, ax = plt.subplots()
for target, label, color in [(0, "Ship", "#263f59"), (1, "Iceberg", "#168376")]:
    values = [float(row["probability"]) for row in validation_rows if int(row["target"]) == target]
    ax.hist(values, bins=np.linspace(0, 1, 16), alpha=0.6, label=label, color=color)
ax.set(title="Inspect prediction distributions on the saved holdout",
       xlabel="Predicted iceberg probability", ylabel="Holdout records")
ax.legend()
fig.tight_layout()
plt.show()""")
markdown("""**Try one controlled change.** Copy the bounded configuration, change one factor
(for example, disable augmentation), and use a new run directory with the same split.
Compare log loss and error patterns. A change that improves one short run is a hypothesis
for further testing; do not claim a general improvement from it alone.

## 7. Generate predictions through the same saved model

Set `ICEBERG_FULL_TEST=1` to run inference over the full authorized test file. This takes
longer than the teaching training profile. Output IDs and order come from the input file;
the CLI also writes a checksum receipt. Creating the file does not submit it to Kaggle.""")
code("""if os.environ.get("ICEBERG_FULL_TEST", "0") == "1":
    prediction_receipt = predict(RUN_DIR / "model.pt", DATA_DIR / "test.json",
                                 RUN_DIR / "submission.csv")
    print(prediction_receipt)
else:
    print("Full-test inference not requested. Set ICEBERG_FULL_TEST=1 or use the predict CLI.")""")
markdown("""## 8. Connect the run to existing MLOps services

The run directory is the local evidence boundary. The optional `iceberg track` command logs
a completed run to an existing MLflow deployment and checks the artifact hashes first.
MLflow can proxy those artifacts to the existing SeaweedFS store. The project deploys no servers.

Docker JupyterLab mounts project files, datasets and working outputs. An optional Compose
override makes an existing client-side SeaweedFS filesystem mount visible at `/shared`.
Prefect and W&B adapters remain future extensions; they are not claimed as implemented here.

For Kaggle, the thin notebook installs the project wheel from an attached code dataset,
uses Kaggle-mounted competition data, and saves a portable run bundle. Kaggle execution
is a separate verification target and does not require private service access.

See [reproduction](../docs/reproduction.md), [migration rationale](../docs/migration.md),
[MLOps boundaries](../docs/mlops.md), and [observed results](../docs/results.md).""")

metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
notebook = nb.v4.new_notebook(cells=cells, metadata=metadata)
(ROOT / "notebooks").mkdir(exist_ok=True)
nb.write(notebook, ROOT / "notebooks/01-method-and-rerun.ipynb")

kaggle_cells = [
    nb.v4.new_markdown_cell("""# Iceberg: offline package-based rerun

Attach the code-wheel dataset named in the metadata and accept the competition rules.
This is a bounded method-port rerun, not a reproduction of the historical leaderboard score.
It trains the package's original-shape CNN on 128 training records, evaluates the full saved
holdout, and predicts the full test input. Internet and private services are not required."""),
    nb.v4.new_code_cell("""from pathlib import Path
import subprocess
import sys

wheel_roots = [p for p in Path("/kaggle/input").rglob("iceberg_classifier-*.whl")]
assert len(wheel_roots) == 1, "Attach exactly one project wheel"
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                str(wheel_roots[0])], check=True)
"""),
    nb.v4.new_code_cell(
        """from iceberg.runtime import train, predict
import json
import zipfile

competition = "statoil-iceberg-classifier-challenge"
candidates = [Path("/kaggle/input/competitions") / competition,
              Path("/kaggle/input") / competition]
data_dir = next(p for p in candidates if (p / "train.json").is_file())
run_dir = Path("/kaggle/working/iceberg-run")
config = """
        + repr(json.loads((ROOT / "configs/bounded.json").read_text()))
        + """
manifest = train(data_dir / "train.json", run_dir, config)
receipt = predict(run_dir / "model.pt", data_dir / "test.json",
                  Path("/kaggle/working/submission.csv"))
with zipfile.ZipFile("/kaggle/working/run-bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in run_dir.iterdir():
        if path.is_file():
            bundle.write(path, path.name)
print(json.dumps({"run_id": manifest["logical_run_id"], "prediction": receipt}, indent=2))
"""
    ),
]
(ROOT / "kaggle").mkdir(exist_ok=True)
nb.write(nb.v4.new_notebook(cells=kaggle_cells, metadata=metadata), ROOT / "kaggle/rerun.ipynb")
(ROOT / "kaggle/kernel-metadata.json").write_text(
    json.dumps(
        {
            "id": "cttsai1985/iceberg-modern-method-rerun",
            "title": "Iceberg modern method rerun",
            "code_file": "rerun.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "competition_sources": ["statoil-iceberg-classifier-challenge"],
            "dataset_sources": ["cttsai1985/iceberg-classifier-code"],
            "kernel_sources": [],
            "model_sources": [],
        },
        indent=2,
    )
    + "\n"
)
print("Authored local walkthrough and offline Kaggle entry point.")
