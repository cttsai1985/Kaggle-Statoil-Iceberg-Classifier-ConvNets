"""Prepare code-only assets locally; never upload, publish or submit."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from iceberg.data import sha256
from iceberg.runtime import source_identity

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path(".local/kaggle-bundle"))
parser.add_argument("--owner", default="cttsai", help="Kaggle username, independent of GitHub")
args = parser.parse_args()
assets = args.output / "code-dataset"
kernel = args.output / "kernel"
assets.mkdir(parents=True, exist_ok=True)
kernel.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="iceberg-wheel-") as temporary:
    subprocess.run(["uv", "build", "--wheel", "--out-dir", temporary], check=True)
    wheels = list(Path(temporary).glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one newly built project wheel")
    # A staging directory represents one code version; remove stale generated wheels only.
    for previous in assets.glob("iceberg_classifier-*.whl"):
        previous.unlink()
    wheel = assets / wheels[0].name
    shutil.copy2(wheels[0], wheel)
shutil.copy2("LICENSE", assets / "LICENSE")
for config in Path("configs").glob("*.json"):
    shutil.copy2(config, assets / ("config-" + config.name))
(assets / "source.json").write_text(
    json.dumps(
        {
            "wheel": wheel.name,
            "sha256": sha256(wheel),
            "source": "https://github.com/cttsai1985/kaggle-iceberg-solution-mlops",
            **source_identity(),
            "lock_sha256": sha256(Path("uv.lock")),
            "config_sha256": {
                "config-" + path.name: sha256(path) for path in sorted(Path("configs").glob("*.json"))
            },
        },
        indent=2,
    )
    + "\n"
)
(assets / "dataset-metadata.json").write_text(
    json.dumps(
        {
            "id": f"{args.owner}/kaggle-iceberg-code",
            "title": "Iceberg classifier code",
            "licenses": [{"name": "MIT"}],
        },
        indent=2,
    )
    + "\n"
)
for name in ["rerun.ipynb", "kernel-metadata.json"]:
    shutil.copy2(Path("notebooks/kaggle") / name, kernel / name)
metadata = json.loads((kernel / "kernel-metadata.json").read_text())
metadata["id"] = f"{args.owner}/kaggle-iceberg-solution-mlops"
metadata["dataset_sources"] = [f"{args.owner}/kaggle-iceberg-code"]
(kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Prepared code-only assets in {args.output}; no Kaggle resources changed.")
