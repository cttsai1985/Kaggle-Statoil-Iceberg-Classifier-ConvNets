"""Modern training, portable inference, and inspectable experiment evidence."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader

from iceberg import PROJECT_ID
from iceberg.data import Preprocessing, RadarDataset, load_data, sha256
from iceberg.model import IcebergCNN

REFERENCE_COMMIT = "d7f9f46145f067a82d3ddf3a9e6ca134bd098083"
ARTIFACTS = {
    "model.pt",
    "config.json",
    "split.csv",
    "history.json",
    "metrics.json",
    "validation.csv",
}
DEFAULTS = {
    "name": "modern-image-baseline",
    "seed": 1017,
    "split_seed": 25,
    "validation_fraction": 0.159,
    "filters": 16,
    "hidden": 64,
    "use_angle": False,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.001,
    "min_learning_rate": 0.00001,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "augment": True,
    "zoom_policy": "current",
    "max_train_rows": None,
    "threads": 4,
    "device": "auto",
    "deterministic": False,
    "mixed_precision": True,
    "tta": "none",
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, columns: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def resolve_config(config: dict) -> dict:
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = DEFAULTS | config
    for key in ["threads", "batch_size", "epochs", "filters", "hidden"]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ["seed", "split_seed"]:
        if type(config[key]) is not int or not 0 <= config[key] < 2**32:
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    for key in ["use_angle", "augment", "deterministic", "mixed_precision"]:
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in [
        "validation_fraction",
        "dropout",
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "gradient_clip",
    ]:
        if not isinstance(config[key], (int, float)) or not np.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 < config["validation_fraction"] < 1 or not 0 <= config["dropout"] < 1:
        raise ValueError("Invalid validation fraction or dropout")
    if (
        not 0 <= config["min_learning_rate"] <= config["learning_rate"]
        or config["learning_rate"] <= 0
    ):
        raise ValueError("Invalid learning rate range")
    if config["weight_decay"] < 0 or config["gradient_clip"] <= 0:
        raise ValueError("Invalid weight decay or gradient clipping")
    limit = config["max_train_rows"]
    if limit is not None and (type(limit) is not int or limit < 2):
        raise ValueError("max_train_rows must be null or an integer >= 2")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if config["zoom_policy"] not in {"current", "none", "mild_5pct", "extreme_5pct"}:
        raise ValueError("Unknown zoom policy")
    if config["tta"] not in {"none", "flips"}:
        raise ValueError("tta must be none or flips")
    return config


def select_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose cpu or auto")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def configure(config: dict) -> torch.device:
    if config["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.benchmark = not config["deterministic"]
    torch.backends.cudnn.deterministic = config["deterministic"]
    return select_device(config["device"])


def source_identity() -> dict:
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())

    def git(*args):
        if not (root / ".git").exists():
            return None
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "legacy_commit": REFERENCE_COMMIT,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python_source_sha256": digest.hexdigest(),
    }


def environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["torch", "numpy", "scipy", "scikit-learn", "iceberg-classifier"]
        },
    }


def build_model(config: dict) -> IcebergCNN:
    return IcebergCNN(
        config["filters"],
        config["hidden"],
        use_angle=config["use_angle"],
        dropout=config["dropout"],
    )


@torch.inference_mode()
def probabilities(model, dataset, batch_size: int, *, tta: str = "none") -> np.ndarray:
    """Evaluate all rows, optionally averaging four paired-flip probabilities."""
    if tta not in {"none", "flips"}:
        raise ValueError("tta must be none or flips")
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    flips = [(), (-1,), (-2,), (-2, -1)] if tta == "flips" else [()]
    for images, angles, _ in DataLoader(dataset, batch_size=batch_size):
        images, angles = images.to(device), angles.to(device)
        prediction = torch.stack(
            [
                torch.sigmoid(model(torch.flip(images, dims) if dims else images, angles))
                for dims in flips
            ]
        ).mean(0)
        outputs.append(prediction.cpu().numpy())
    if not outputs:
        raise ValueError("Cannot predict an empty dataset")
    result = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise ValueError("Invalid model probabilities")
    return result


def angle_overlap(data, train_idx, valid_idx) -> dict:
    """Audit repeated acquisition angles without using labels or test records."""
    fit = data.angles[train_idx, 0]
    holdout = data.angles[valid_idx, 0]
    known_fit = set(fit[np.isfinite(fit)].tolist())
    known_valid = set(holdout[np.isfinite(holdout)].tolist())
    return {
        "shared_known_angles": len(known_fit & known_valid),
        "validation_rows_with_training_angle": int(sum(float(a) in known_fit for a in holdout)),
        "training_missing_angles": int(np.isnan(fit).sum()),
        "validation_missing_angles": int(np.isnan(holdout).sum()),
    }


def validate_split_indices(rows: int, split_indices):
    """Require a complete, disjoint partition before fitting any transformations."""
    if len(split_indices) != 2:
        raise ValueError("Expected training and validation index arrays")
    indices = tuple(np.asarray(part) for part in split_indices)
    for part in indices:
        if part.ndim != 1 or part.dtype.kind not in "iu" or not len(part):
            raise ValueError("Split indices must be nonempty one-dimensional integer arrays")
        if len(np.unique(part)) != len(part) or np.any(part < 0) or np.any(part >= rows):
            raise ValueError("Split indices contain duplicates or out-of-range rows")
    if np.intersect1d(*indices).size:
        raise ValueError("Training and validation indices overlap")
    if len(indices[0]) < 2 or len(indices[0]) + len(indices[1]) != rows:
        raise ValueError("Explicit split must cover every row with at least two training rows")
    return indices


def train(
    data_path: Path,
    run_dir: Path,
    config: dict,
    *,
    fixture=False,
    split_indices=None,
    checkpoint_selection: str = "best",
    split_metadata: dict | None = None,
) -> dict:
    config = resolve_config(config)
    if checkpoint_selection not in {"best", "last"}:
        raise ValueError("checkpoint_selection must be best or last")
    if split_indices is not None and config["max_train_rows"] is not None:
        raise ValueError("Explicit folds cannot be combined with max_train_rows")
    device = configure(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 2,
        "project_id": PROJECT_ID,
        "logical_run_id": str(uuid.uuid4()),
        "status": "running",
        "run_kind": "synthetic_fixture" if fixture else "real_data_model_run",
        "profile": config["name"],
        "source": source_identity(),
        "environment": environment(device),
        "config": config,
        "input_sha256": sha256(data_path),
        "historical_public_target": 0.19348,
        "historical_score_comparison": "requires comparable Kaggle submission score",
        "model_family": "modern-two-band-cnn-v1",
        "checkpoint_selection": checkpoint_selection,
        "split_metadata": split_metadata,
    }
    write_json(run_dir / "config.json", config)
    manifest["config_sha256"] = sha256(run_dir / "config.json")
    write_json(run_dir / "manifest.json", manifest)
    try:
        data = load_data(data_path, labeled=True)
        if split_indices is None:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=config["validation_fraction"],
                random_state=config["split_seed"],
            )
            train_idx, valid_idx = next(splitter.split(data.images, data.targets))
        else:
            train_idx, valid_idx = validate_split_indices(len(data.ids), split_indices)
        limit = config["max_train_rows"]
        if limit is not None and limit < len(train_idx):
            train_idx, _ = train_test_split(
                train_idx,
                train_size=limit,
                stratify=data.targets[train_idx],
                random_state=config["seed"],
            )
        fit_ids, holdout_ids = set(train_idx.tolist()), set(valid_idx.tolist())
        write_csv(
            run_dir / "split.csv",
            ["id", "role"],
            (
                (
                    identifier,
                    "train" if i in fit_ids else "validation" if i in holdout_ids else "unused",
                )
                for i, identifier in enumerate(data.ids)
            ),
        )
        preprocessing = Preprocessing.fit(data, train_idx)
        manifest.update(
            split_sha256=sha256(run_dir / "split.csv"),
            preprocessing=preprocessing.to_dict(),
            split_audit=angle_overlap(data, train_idx, valid_idx),
            train_rows=len(train_idx),
            validation_rows=len(valid_idx),
        )
        write_json(run_dir / "manifest.json", manifest)
        train_set = RadarDataset(
            data,
            train_idx,
            preprocessing,
            augment_images=config["augment"],
            seed=config["seed"],
            zoom_policy=config["zoom_policy"],
        )
        validation = RadarDataset(data, valid_idx, preprocessing)
        loader = DataLoader(
            train_set,
            batch_size=min(config["batch_size"], len(train_idx)),
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        model = build_model(config).to(device)
        # Scalars/vectors include biases and normalization weights, which are not decayed.
        decay, no_decay = [], []
        for parameter in model.parameters():
            (no_decay if parameter.ndim <= 1 else decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": config["weight_decay"]},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config["learning_rate"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["min_learning_rate"]
        )
        use_amp = config["mixed_precision"] and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        loss_fn = nn.BCEWithLogitsLoss()
        history, best_loss = [], float("inf")
        examples_seen, optimizer_steps = 0, 0
        for epoch in range(1, config["epochs"] + 1):
            epoch_started = time.monotonic()
            model.train()
            loss_sum, epoch_rows = 0.0, 0
            rate = optimizer.param_groups[0]["lr"]
            for images, angles, targets in loader:
                images, angles = images.to(device), angles.to(device)
                targets = targets.float().to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = loss_fn(model(images, angles), targets)
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is non-finite")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += int(scaler.get_scale() >= scale_before)
                count = len(targets)
                examples_seen += count
                epoch_rows += count
                loss_sum += float(loss.detach()) * count
            prediction = probabilities(model, validation, config["batch_size"], tta=config["tta"])
            score = float(log_loss(data.targets[valid_idx], prediction, labels=[0, 1]))
            history.append(
                {
                    "epoch": epoch,
                    "training_log_loss": loss_sum / epoch_rows,
                    "validation_log_loss": score,
                    "learning_rate": rate,
                    "optimizer_steps": optimizer_steps,
                    "training_examples_seen": examples_seen,
                    "epoch_seconds": time.monotonic() - epoch_started,
                }
            )
            print(json.dumps(history[-1]), flush=True)
            if checkpoint_selection == "last" or score < best_loss:
                best_loss = score
                torch.save(
                    {
                        "schema_version": 2,
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config,
                        "preprocessing": preprocessing.to_dict(),
                        "validation_indices": valid_idx.tolist(),
                        "input_sha256": manifest["input_sha256"],
                        "split_sha256": manifest["split_sha256"],
                        "epoch": epoch,
                        "validation_log_loss": score,
                        "checkpoint_selection": checkpoint_selection,
                    },
                    run_dir / "model.pt",
                )
            scheduler.step()
            write_json(run_dir / "history.json", history)
        metrics = evaluate(
            run_dir / "model.pt", data_path, run_dir / "validation.csv", device=str(device)
        )
        write_json(run_dir / "metrics.json", metrics)
        manifest.update(
            status="completed",
            training_examples_seen=examples_seen,
            optimizer_steps=optimizer_steps,
            steps_per_epoch=len(loader),
            elapsed_seconds=time.monotonic() - started,
            artifacts={name: sha256(run_dir / name) for name in sorted(ARTIFACTS)},
        )
        write_json(run_dir / "manifest.json", manifest)
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(run_dir / "manifest.json", manifest)
        raise


def load_model(checkpoint: Path, *, device: str = "auto"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_version") != 2:
        raise ValueError("Unsupported checkpoint: rerun the modern training pipeline")
    config = resolve_config(saved["config"])
    torch.set_num_threads(config["threads"])
    model = build_model(config)
    model.load_state_dict(saved["model"], strict=True)
    model.to(select_device(device)).eval()
    return model, saved


def evaluate(
    checkpoint: Path,
    data_path: Path,
    output: Path | None = None,
    *,
    device: str = "auto",
    tta: str | None = None,
) -> dict:
    model, saved = load_model(checkpoint, device=device)
    if sha256(data_path) != saved["input_sha256"]:
        raise ValueError("Evaluation input differs from the saved training/split input")
    data = load_data(data_path, labeled=True)
    indices = saved["validation_indices"]
    preprocessing = Preprocessing(**saved["preprocessing"])
    prediction = probabilities(
        model,
        RadarDataset(data, indices, preprocessing),
        saved["config"]["batch_size"],
        tta=saved["config"]["tta"] if tta is None else tta,
    )
    targets = data.targets[indices]
    if output is not None:
        write_csv(
            output,
            ["id", "target", "probability"],
            (
                (data.ids[i], int(y), float(p))
                for i, y, p in zip(indices, targets, prediction, strict=True)
            ),
        )
    metrics = {
        "validation_log_loss": float(log_loss(targets, prediction, labels=[0, 1])),
        "validation_rows": len(indices),
        "brier_score": float(np.mean((targets - prediction) ** 2)),
        "probability_min": float(prediction.min()),
        "probability_max": float(prediction.max()),
    }
    if saved.get("checkpoint_selection", "best") == "best":
        metrics["best_epoch"] = saved["epoch"]
    else:
        metrics["selected_epoch"] = saved["epoch"]
    missing = np.isnan(data.angles[indices, 0])
    for name, mask in [("missing_angle", missing), ("known_angle", ~missing)]:
        if mask.any():
            metrics[f"{name}_log_loss"] = float(
                log_loss(targets[mask], prediction[mask], labels=[0, 1])
            )
    return metrics


def predict(
    checkpoint: Path, data_path: Path, output: Path, *, device: str = "auto", tta: str | None = None
) -> dict:
    model, saved = load_model(checkpoint, device=device)
    data = load_data(data_path, labeled=False)
    selected_tta = saved["config"]["tta"] if tta is None else tta
    prediction = probabilities(
        model,
        RadarDataset(data, range(len(data.ids)), Preprocessing(**saved["preprocessing"])),
        saved["config"]["batch_size"],
        tta=selected_tta,
    )
    write_csv(output, ["id", "is_iceberg"], zip(data.ids, prediction, strict=True))
    receipt = {
        "rows": len(data.ids),
        "input_sha256": sha256(data_path),
        "checkpoint_sha256": sha256(checkpoint),
        "output_sha256": sha256(output),
        "checkpoint_profile": saved["config"]["name"],
        "tta": selected_tta,
        "status": "model_predictions",
    }
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def verify_run(run_dir: Path, data_path: Path | None = None) -> dict:
    """Verify a completed local/Kaggle bundle before reusing its results."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != 2 or manifest.get("status") != "completed":
        raise ValueError("Expected a completed modern run")
    if set(manifest.get("artifacts", {})) != ARTIFACTS:
        raise ValueError("Unexpected artifact manifest")
    for filename, digest in manifest["artifacts"].items():
        path = run_dir / filename
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Artifact checksum mismatch: {filename}")
    if manifest["config_sha256"] != sha256(run_dir / "config.json"):
        raise ValueError("Configuration fingerprint differs")
    if manifest["split_sha256"] != sha256(run_dir / "split.csv"):
        raise ValueError("Split fingerprint differs")
    if json.loads((run_dir / "config.json").read_text()) != manifest["config"]:
        raise ValueError("Manifest configuration differs")
    if data_path is not None and sha256(data_path) != manifest["input_sha256"]:
        raise ValueError("Run input differs from the supplied training data")
    return manifest
