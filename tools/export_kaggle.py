"""Prepare code-only assets locally; never upload, publish or submit."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from iceberg.data import sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("kaggle-bundle"))
args = parser.parse_args()
subprocess.run(["uv", "build", "--wheel"], check=True)
wheel = next(Path("dist").glob("iceberg_classifier-*.whl"))
assets = args.output / "code-dataset"
kernel = args.output / "kernel"
assets.mkdir(parents=True, exist_ok=True)
kernel.mkdir(parents=True, exist_ok=True)
shutil.copy2(wheel, assets / wheel.name)
shutil.copy2("LICENSE", assets / "LICENSE")
(assets / "source.json").write_text(
    json.dumps(
        {
            "wheel": wheel.name,
            "sha256": sha256(wheel),
            "source": "https://github.com/cttsai1985/Kaggle-Statoil-Iceberg-Classifier-ConvNets",
        },
        indent=2,
    )
    + "\n"
)
(assets / "dataset-metadata.json").write_text(
    json.dumps(
        {
            "id": "cttsai1985/iceberg-classifier-code",
            "title": "Iceberg classifier code",
            "licenses": [{"name": "MIT"}],
        },
        indent=2,
    )
    + "\n"
)
for name in ["rerun.ipynb", "kernel-metadata.json"]:
    shutil.copy2(Path("kaggle") / name, kernel / name)
print(f"Prepared code-only assets in {args.output}; no Kaggle resources changed.")
