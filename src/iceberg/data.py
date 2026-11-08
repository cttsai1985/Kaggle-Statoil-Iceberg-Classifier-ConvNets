"""Explicit data boundaries shared by the CLI, notebooks, and Kaggle runtime."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import affine_transform
from torch.utils.data import Dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RadarData:
    ids: list[str]
    images: np.ndarray
    angles: np.ndarray
    targets: np.ndarray | None


def load_data(path: Path, *, labeled: bool) -> RadarData:
    """Read official JSON, preserving row order, scaling and the -1 angle sentinel."""
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON list")
    ids, images, angles, targets = [], [], [], []
    for row in records:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every record needs a nonempty string id")
        bands = np.asarray([row["band_1"], row["band_2"]], dtype=np.float32)
        if bands.shape != (2, 75 * 75) or not np.isfinite(bands).all():
            raise ValueError(f"Invalid radar bands for {identifier}")
        angle = -1.0 if row["inc_angle"] == "na" else float(row["inc_angle"])
        if not np.isfinite(angle):
            raise ValueError(f"Non-finite incidence angle for {identifier}")
        ids.append(identifier)
        images.append(bands.reshape(2, 75, 75) / 100.0 + 0.5)
        angles.append([angle])
        if labeled:
            target = row["is_iceberg"]
            if target not in (0, 1):
                raise ValueError("Targets must be binary")
            targets.append(target)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record ids")
    return RadarData(
        ids,
        np.stack(images),
        np.asarray(angles, dtype=np.float32),
        np.asarray(targets, dtype=np.float32) if labeled else None,
    )


def spatial_affine(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Keras-style centered, first-order, nearest-fill transform applied to both bands."""
    center = np.asarray(image.shape[-2:], dtype=float) / 2.0 + 0.5
    offset = center - matrix @ center
    return np.stack(
        [affine_transform(band, matrix, offset, order=1, mode="nearest") for band in image]
    )


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = image.copy()
    if rng.random() < 0.1:
        radians = np.deg2rad(rng.uniform(-45, 45))
        image = spatial_affine(
            image,
            np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]),
        )
    if rng.random() < 0.1:
        image = np.rot90(image, k=int(rng.integers(0, 4)), axes=(1, 2))
    if rng.random() < 0.05:
        displacement = rng.uniform(-0.1, 0.1, size=2) * np.asarray(image.shape[-2:])
        image = np.stack(
            [
                affine_transform(band, np.eye(2), displacement, order=1, mode="nearest")
                for band in image
            ]
        )
    if rng.random() < 0.05:
        # The original uses zoom_range=(0.1, 0.1), not the common (0.9, 1.1).
        image = spatial_affine(image, np.eye(2) * 0.1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=2)
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
    return np.ascontiguousarray(image)


class RadarDataset(Dataset):
    def __init__(self, data: RadarData, indices, *, augment_images=False, seed=1017):
        self.data = data
        self.indices = np.asarray(indices)
        self.augment_images = augment_images
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        image = self.data.images[index]
        if self.augment_images:
            image = augment(image, self.rng)
        target = self.data.targets[index] if self.data.targets is not None else 0.0
        return torch.from_numpy(image), torch.from_numpy(self.data.angles[index]), target


def make_fixture(directory: Path, rows: int = 32) -> None:
    """Small labeled synthetic fixture for pipeline checks; never competition evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    records = [
        {
            "id": f"fixture-{i}",
            "band_1": rng.normal(-20, 2, 75 * 75).tolist(),
            "band_2": rng.normal(-25, 2, 75 * 75).tolist(),
            "inc_angle": "na" if i == 0 else 35.0 + i / 100,
            "is_iceberg": i % 2,
        }
        for i in range(rows)
    ]
    (directory / "train.json").write_text(json.dumps(records))
    (directory / "test.json").write_text(
        json.dumps(
            [
                {**{k: v for k, v in row.items() if k != "is_iceberg"}, "id": f"test-fixture-{i}"}
                for i, row in enumerate(records[:8])
            ]
        )
    )
