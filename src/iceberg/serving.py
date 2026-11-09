"""Package a verified two-band checkpoint for MLflow's prediction interface.

The serving model owns the fitted preprocessing and recorded TTA policy. Export
never fits a model or copies row-level predictions into the package.
"""

import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from mlflow.types.schema import Array, DataType

from iceberg.data import Preprocessing, RadarData, RadarDataset, sha256
from iceberg.runtime import load_model, probabilities, source_identity, verify_run


def input_signature() -> ModelSignature:
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(Array(DataType.double), "band_1"),
                ColSpec(Array(DataType.double), "band_2"),
                ColSpec(DataType.double, "inc_angle", required=False),
            ]
        ),
        outputs=Schema([ColSpec(DataType.double, "is_iceberg")]),
    )


def frame_from_records(records: list[dict]) -> pd.DataFrame:
    """Select inference fields without carrying identifiers or targets into the model."""
    return pd.DataFrame(
        {
            "band_1": [row["band_1"] for row in records],
            "band_2": [row["band_2"] for row in records],
            "inc_angle": [
                np.nan if row.get("inc_angle") in (None, "na") else float(row["inc_angle"])
                for row in records
            ],
        }
    )


def radar_from_frame(frame: pd.DataFrame, max_batch_rows: int) -> RadarData:
    if not isinstance(frame, pd.DataFrame) or not 1 <= len(frame) <= max_batch_rows:
        raise ValueError(f"Provide between 1 and {max_batch_rows} radar records")
    if set(frame) - {"band_1", "band_2", "inc_angle"} or not {"band_1", "band_2"} <= set(frame):
        raise ValueError("Expected band_1, band_2 and optional inc_angle columns")
    images = []
    for band in ["band_1", "band_2"]:
        try:
            values = np.asarray(frame[band].tolist(), dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("Radar bands must contain numeric arrays") from error
        if values.shape != (len(frame), 75 * 75) or not np.isfinite(values).all():
            raise ValueError("Each radar band must contain 5625 finite values")
        images.append(values.reshape(-1, 75, 75))
    angles = (
        np.asarray(frame["inc_angle"], dtype=np.float64)
        if "inc_angle" in frame
        else np.full(len(frame), np.nan)
    )
    if np.isinf(angles).any():
        raise ValueError("Incidence angle must be finite or missing")
    return RadarData(
        [str(index) for index in range(len(frame))],
        np.stack(images, axis=1),
        angles[:, None],
        None,
    )


class IcebergModel(mlflow.pyfunc.PythonModel):
    def __init__(self, *, checkpoint_sha256, manifest_sha256, threads=2, max_batch_rows=64):
        if threads not in {1, 2, 4} or not 1 <= max_batch_rows <= 256:
            raise ValueError("Use 1, 2 or 4 CPU threads and a batch limit between 1 and 256")
        self.checkpoint_sha256 = checkpoint_sha256
        self.manifest_sha256 = manifest_sha256
        self.threads = threads
        self.max_batch_rows = max_batch_rows

    def load_context(self, context):
        checkpoint = Path(context.artifacts["checkpoint"])
        manifest_path = Path(context.artifacts["producer_manifest"])
        if (
            sha256(checkpoint) != self.checkpoint_sha256
            or sha256(manifest_path) != self.manifest_sha256
        ):
            raise ValueError("Serving model differs from its verified producer artifacts")
        manifest = json.loads(manifest_path.read_text())
        if manifest["artifacts"]["model.pt"] != self.checkpoint_sha256:
            raise ValueError("Producer manifest does not bind this checkpoint")
        self.model, self.saved = load_model(checkpoint, device="cpu")
        if self.saved["config"] != manifest["config"]:
            raise ValueError("Checkpoint configuration differs from producer manifest")
        torch.set_num_threads(self.threads)
        self.preprocessing = Preprocessing(**self.saved["preprocessing"])

    def predict(self, context, model_input, params=None):
        data = radar_from_frame(model_input, self.max_batch_rows)
        prediction = probabilities(
            self.model,
            RadarDataset(data, range(len(data.ids)), self.preprocessing),
            batch_size=min(self.max_batch_rows, self.saved["config"]["batch_size"]),
            tta=self.saved["config"]["tta"],
        )
        return pd.DataFrame({"is_iceberg": prediction}, index=model_input.index)


def export_model(run_dir: Path, output: Path, *, threads=2, max_batch_rows=64) -> dict:
    """Create a standalone package from a verified producer; refuse replacement."""
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("Use a new model-package directory; existing packages are immutable")
    manifest = verify_run(run_dir)
    manifest_sha = sha256(run_dir / "manifest.json")
    checkpoint_sha = manifest["artifacts"]["model.pt"]
    adapter = IcebergModel(
        checkpoint_sha256=checkpoint_sha,
        manifest_sha256=manifest_sha,
        threads=threads,
        max_batch_rows=max_batch_rows,
    )
    requirements = ["--extra-index-url https://download.pytorch.org/whl/cpu"] + [
        f"{name}=={importlib.metadata.version(name)}"
        for name in ["mlflow", "numpy", "pandas", "scipy", "scikit-learn", "torch"]
    ]
    # Synthetic example documents the request without publishing competition pixels.
    example = pd.DataFrame(
        {"band_1": [[0.0] * 5625], "band_2": [[0.0] * 5625], "inc_angle": [35.0]}
    )
    with tempfile.TemporaryDirectory() as directory:
        package = Path(directory) / "iceberg"
        shutil.copytree(
            Path(__file__).parent,
            package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        mlflow.pyfunc.save_model(
            path=str(output),
            python_model=adapter,
            artifacts={
                "checkpoint": str(run_dir / "model.pt"),
                "producer_manifest": str(run_dir / "manifest.json"),
            },
            code_paths=[str(package)],
            signature=input_signature(),
            input_example=example,
            pip_requirements=requirements,
            metadata={"project_id": "iceberg", "producer_manifest_sha256": manifest_sha},
        )
    # Model and copied source are read-only at rest; runtime loads never refit them.
    for tree in [output / "artifacts", output / "code"]:
        for item in tree.rglob("*"):
            item.chmod(0o555 if item.is_dir() else 0o444)
        tree.chmod(0o555)
    receipt = {
        "schema_version": 1,
        "project_id": "iceberg",
        "status": "mlflow_model_packaged",
        "producer_run_kind": manifest["run_kind"],
        "producer_logical_run_id": manifest["logical_run_id"],
        "producer_source": manifest["source"],
        "producer_manifest_sha256": manifest_sha,
        "checkpoint_sha256": checkpoint_sha,
        "packaging_source": source_identity(),
        "inference": {
            "device": "cpu",
            "threads": threads,
            "max_batch_rows": max_batch_rows,
            "tta": manifest["config"]["tta"],
            "radar_shape": [2, 75, 75],
            "angle_missingness": "nullable_or_omitted",
        },
        "competition_data_included": False,
        "row_level_predictions_included": False,
    }
    (output / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
