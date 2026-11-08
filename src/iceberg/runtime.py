"""Bounded training, independent evaluation, and portable evidence."""

import csv
import hashlib
import importlib.metadata
import json
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

from iceberg.data import RadarDataset, load_data, sha256
from iceberg.model import IcebergCNN

REFERENCE_COMMIT = "d7f9f46145f067a82d3ddf3a9e6ca134bd098083"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, columns: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def configure(config: dict) -> None:
    if config["threads"] < 1 or config["batch_size"] < 1 or config["epochs"] < 1:
        raise ValueError("Threads, batch size and epochs must be positive")
    if config["epoch_repeats"] < 1 or config["filters"] < 1 or config["hidden"] < 1:
        raise ValueError("Repeats and model dimensions must be positive")
    if not 0 < config["validation_fraction"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Invalid validation fraction or learning rate")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "legacy_commit": REFERENCE_COMMIT,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python_source_sha256": digest.hexdigest(),
    }


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["torch", "numpy", "scipy", "scikit-learn", "iceberg-classifier"]
        },
    }


@torch.inference_mode()
def probabilities(model, dataset, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = [
        torch.sigmoid(model(images, angles)).numpy()
        for images, angles, _ in DataLoader(dataset, batch_size=batch_size)
    ]
    result = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise ValueError("Invalid model probabilities")
    return result


def train(data_path: Path, run_dir: Path, config: dict, *, fixture=False) -> dict:
    configure(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 1,
        "logical_run_id": str(uuid.uuid4()),
        "status": "running",
        "run_kind": "synthetic_fixture" if fixture else "real_data_model_run",
        "profile": config["name"],
        "source": source_identity(),
        "environment": environment(),
        "config": config,
        "input_sha256": sha256(data_path),
        "historical_score_comparison": "not established",
        "framework_equivalence": "structural port; numerical equivalence not established",
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "config.json", config)
    try:
        data = load_data(data_path, labeled=True)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=config["validation_fraction"], random_state=config["split_seed"]
        )
        train_idx, valid_idx = next(splitter.split(data.images, data.targets))
        limit = config["max_train_rows"]
        if limit is not None:
            if not isinstance(limit, int) or limit < 2:
                raise ValueError("max_train_rows must be null or an integer >= 2")
            if limit < len(train_idx):
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
        train_set = RadarDataset(
            data, train_idx, augment_images=config["augment"], seed=config["seed"]
        )
        validation = RadarDataset(data, valid_idx)
        loader = DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        model = IcebergCNN(config["filters"], config["hidden"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-8)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0 / (1.0 + config["learning_rate_decay"] * step)
        )
        loss_fn = nn.BCEWithLogitsLoss()
        history, best_loss = [], float("inf")
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            losses = []
            for _ in range(config["epoch_repeats"]):
                for images, angles, targets in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(images, angles), targets.float())
                    if not torch.isfinite(loss):
                        raise ValueError("Training loss is non-finite")
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    losses.append(float(loss.detach()))
            prediction = probabilities(model, validation, config["batch_size"])
            score = float(log_loss(data.targets[valid_idx], prediction, labels=[0, 1]))
            history.append(
                {
                    "epoch": epoch,
                    "training_batch_loss_mean": float(np.mean(losses)),
                    "validation_log_loss": score,
                }
            )
            print(json.dumps(history[-1]), flush=True)
            if score < best_loss:
                best_loss = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config,
                        "validation_indices": valid_idx.tolist(),
                        "input_sha256": manifest["input_sha256"],
                    },
                    run_dir / "model.pt",
                )
            write_json(run_dir / "history.json", history)
        metrics = evaluate(run_dir / "model.pt", data_path, run_dir / "validation.csv")
        write_json(run_dir / "metrics.json", metrics)
        manifest.update(
            status="completed",
            train_rows=len(train_idx),
            validation_rows=len(valid_idx),
            training_examples_seen=len(train_idx) * config["epochs"] * config["epoch_repeats"],
            elapsed_seconds=time.monotonic() - started,
            artifacts={
                name: sha256(run_dir / name)
                for name in [
                    "model.pt",
                    "config.json",
                    "split.csv",
                    "history.json",
                    "metrics.json",
                    "validation.csv",
                ]
            },
        )
        write_json(run_dir / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(run_dir / "manifest.json", manifest)
        raise


def load_model(checkpoint: Path):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    configure(saved["config"])
    model = IcebergCNN(saved["config"]["filters"], saved["config"]["hidden"])
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, saved


def evaluate(checkpoint: Path, data_path: Path, output: Path | None = None) -> dict:
    model, saved = load_model(checkpoint)
    if sha256(data_path) != saved["input_sha256"]:
        raise ValueError("Evaluation input differs from the saved training/split input")
    data = load_data(data_path, labeled=True)
    indices = saved["validation_indices"]
    prediction = probabilities(model, RadarDataset(data, indices), saved["config"]["batch_size"])
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
    return {
        "validation_log_loss": float(log_loss(targets, prediction, labels=[0, 1])),
        "validation_rows": len(indices),
    }


def predict(checkpoint: Path, data_path: Path, output: Path) -> dict:
    model, saved = load_model(checkpoint)
    data = load_data(data_path, labeled=False)
    prediction = probabilities(
        model, RadarDataset(data, range(len(data.ids))), saved["config"]["batch_size"]
    )
    write_csv(output, ["id", "is_iceberg"], zip(data.ids, prediction, strict=True))
    receipt = {
        "rows": len(data.ids),
        "input_sha256": sha256(data_path),
        "checkpoint_sha256": sha256(checkpoint),
        "output_sha256": sha256(output),
        "checkpoint_profile": saved["config"]["name"],
        "status": "model_predictions",
    }
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt
