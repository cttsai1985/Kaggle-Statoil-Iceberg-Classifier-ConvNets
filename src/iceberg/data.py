"""Explicit data boundaries shared by the CLI, notebooks, and Kaggle runtime."""

import hashlib
import json
from dataclasses import asdict, dataclass
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
    """Read raw decibel bands in row order, representing missing angles as NaN."""
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
        missing_angle = row["inc_angle"] in ("na", None)
        angle = float("nan") if missing_angle else float(row["inc_angle"])
        if not missing_angle and not np.isfinite(angle):
            raise ValueError(f"Non-finite incidence angle for {identifier}")
        ids.append(identifier)
        images.append(bands.reshape(2, 75, 75))
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
        np.asarray(angles, dtype=np.float64),
        np.asarray(targets, dtype=np.float32) if labeled else None,
    )


@dataclass(frozen=True)
class Preprocessing:
    """Serializable fit-only statistics; inference never fits new statistics."""

    channel_mean: list[float]
    channel_std: list[float]
    angle_median: float
    angle_mean: float
    angle_std: float

    @classmethod
    def fit(cls, data: RadarData, indices):
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            raise ValueError("Cannot fit preprocessing without training rows")
        images = data.images[indices]
        mean = images.mean(axis=(0, 2, 3), dtype=np.float64)
        std = images.std(axis=(0, 2, 3), dtype=np.float64).clip(min=1e-6)
        angles = data.angles[indices, 0]
        known = angles[np.isfinite(angles)].astype(np.float64)
        median = float(np.median(known)) if len(known) else 0.0
        imputed = np.where(np.isfinite(angles), angles, median).astype(np.float64)
        return cls(
            mean.tolist(),
            std.tolist(),
            median,
            float(imputed.mean()),
            max(float(imputed.std()), 1e-6),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def image(self, image: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.channel_mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.channel_std, dtype=np.float32)[:, None, None]
        return np.ascontiguousarray((image - mean) / std)

    def metadata(self, angles: np.ndarray) -> np.ndarray:
        values = angles.reshape(-1)
        missing = ~np.isfinite(values)
        imputed = np.where(missing, self.angle_median, values)
        return np.column_stack([(imputed - self.angle_mean) / self.angle_std, missing]).astype(
            np.float32
        )


def spatial_affine(
    image: np.ndarray, matrix: np.ndarray, displacement: np.ndarray | None = None
) -> np.ndarray:
    """One centered bilinear transform shared by both polarization channels."""
    center = (np.asarray(image.shape[-2:], dtype=float) - 1.0) / 2.0
    offset = center - matrix @ center
    if displacement is not None:
        offset += displacement
    return np.stack(
        [affine_transform(band, matrix, offset, order=1, mode="reflect") for band in image]
    )


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = image.copy()
    if rng.random() < 0.5:
        radians = np.deg2rad(rng.uniform(-15, 15))
        inverse_scale = 1.0 / rng.uniform(0.9, 1.1)
        image = spatial_affine(
            image,
            inverse_scale
            * np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]),
            rng.uniform(-0.05, 0.05, size=2) * np.asarray(image.shape[-2:]),
        )
    image = np.rot90(image, k=int(rng.integers(0, 4)), axes=(1, 2))
    if rng.random() < 0.5:
        image = np.flip(image, axis=2)
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
    return np.ascontiguousarray(image)


class RadarDataset(Dataset):
    def __init__(
        self,
        data: RadarData,
        indices,
        preprocessing: Preprocessing,
        *,
        augment_images=False,
        seed=1017,
    ):
        self.data = data
        self.indices = np.asarray(indices)
        self.augment_images = augment_images
        self.rng = np.random.default_rng(seed)
        self.preprocessing = preprocessing
        self.angles = preprocessing.metadata(data.angles)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        image = self.data.images[index]
        if self.augment_images:
            image = augment(image, self.rng)
        image = self.preprocessing.image(image)
        target = self.data.targets[index] if self.data.targets is not None else 0.0
        return torch.from_numpy(image), torch.from_numpy(self.angles[index]), target


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
