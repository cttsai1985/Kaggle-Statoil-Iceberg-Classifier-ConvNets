"""Explicit commands; publishing and competition submission are never automatic."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from iceberg import COMPETITION
from iceberg.data import load_data, make_fixture, sha256
from iceberg.runtime import evaluate, predict, train, verify_run, write_json


def download(directory: Path) -> dict:
    import py7zr

    sibling = Path(sys.executable).with_name("kaggle")
    executable = str(sibling) if sibling.is_file() else shutil.which("kaggle")
    if executable is None:
        raise RuntimeError("Install the data dependency group to use Kaggle CLI")
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN"):
        auth = subprocess.run(
            [executable, "auth", "print-access-token"], capture_output=True, text=True
        )
        if auth.returncode or not auth.stdout.strip() or "\n" in auth.stdout.strip():
            raise RuntimeError("Authenticate with `kaggle auth login` and accept competition rules")
        env["KAGGLE_API_TOKEN"] = auth.stdout.strip()
    directory.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for filename in ["train.json", "test.json", "sample_submission.csv"]:
        archive_path = directory / (filename + ".7z")
        if not archive_path.exists():
            # Capture errors so credentials returned by an unexpected client are never printed.
            result = subprocess.run(
                [
                    executable,
                    "competitions",
                    "download",
                    "-c",
                    COMPETITION,
                    "-f",
                    archive_path.name,
                    "-p",
                    str(directory),
                    "-q",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(
                    f"Kaggle download failed for {filename}; check login/rule access"
                )
        destination = directory / filename
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=directory) as staging:
                with py7zr.SevenZipFile(archive_path) as archive:
                    names = archive.getnames()
                    if any(
                        PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts
                        for n in names
                    ):
                        raise ValueError("Unsafe archive path")
                    archive.extractall(staging)
                candidates = list(Path(staging).rglob(filename))
                if len(candidates) != 1 or candidates[0].is_symlink():
                    raise ValueError(f"Expected exactly one regular {filename} in archive")
                shutil.copyfile(candidates[0], destination)
        receipts[filename] = {"sha256": sha256(destination), "archive_sha256": sha256(archive_path)}
    write_json(
        directory / "download-manifest.json", {"competition": COMPETITION, "files": receipts}
    )
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["download", "validate", "fixture"]:
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path, required=True)
    sub = commands.add_parser("train")
    sub.add_argument("--data-dir", type=Path, required=True)
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--config", type=Path, required=True)
    sub.add_argument("--fixture", action="store_true", help="Label synthetic runs explicitly")
    sub.add_argument("--device", choices=["auto", "cpu", "cuda"])
    sub = commands.add_parser("cv")
    sub.add_argument("--data-dir", type=Path, required=True)
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--config", type=Path, required=True)
    sub.add_argument("--folds", type=int, default=5)
    sub.add_argument("--strategy", choices=["stratified", "grouped"], default="stratified")
    sub.add_argument("--split-seed", type=int, default=25)
    sub.add_argument("--seeds", type=int, nargs="+", default=[1017, 2027])
    sub.add_argument("--workers", type=int, default=1)
    sub.add_argument("--fixture", action="store_true")
    sub.add_argument("--device", choices=["auto", "cpu", "cuda"])
    sub = commands.add_parser("analyze")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--data", type=Path)
    sub = commands.add_parser("predict-cv")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--data", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for name in ["evaluate", "predict"]:
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=name == "predict")
        sub.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        sub.add_argument("--tta", choices=["none", "flips"])
    sub = commands.add_parser("verify")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--data", type=Path)
    for name in ["track", "track-cv"]:
        sub = commands.add_parser(name)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
        sub.add_argument(
            "--resume",
            action="store_true",
            help="Resume an incomplete import after its previous writer has stopped",
        )
    args = parser.parse_args()
    if args.command == "fixture":
        make_fixture(args.data_dir)
        result = {"status": "synthetic_fixture_created"}
    elif args.command == "download":
        result = download(args.data_dir)
    elif args.command == "validate":
        training = load_data(args.data_dir / "train.json", labeled=True)
        testing = load_data(args.data_dir / "test.json", labeled=False)
        if set(training.ids) & set(testing.ids):
            raise ValueError("Training and test ids overlap")
        result = {"train_rows": len(training.ids), "test_rows": len(testing.ids)}
    elif args.command == "train":
        config = json.loads(args.config.read_text())
        if args.device is not None:
            config["device"] = args.device
        result = train(
            args.data_dir / "train.json",
            args.run_dir,
            config,
            fixture=args.fixture,
        )
    elif args.command == "evaluate":
        result = evaluate(args.checkpoint, args.data, args.output, device=args.device, tta=args.tta)
    elif args.command == "cv":
        from iceberg.cv import cross_validate

        config = json.loads(args.config.read_text())
        if args.device is not None:
            config["device"] = args.device
        result = cross_validate(
            args.data_dir / "train.json",
            args.run_dir,
            config,
            folds=args.folds,
            strategy=args.strategy,
            split_seed=args.split_seed,
            seeds=args.seeds,
            workers=args.workers,
            fixture=args.fixture,
        )
    elif args.command == "analyze":
        from iceberg.cv import analyze_cv

        result = analyze_cv(args.run_dir, args.data)
    elif args.command == "predict":
        result = predict(args.checkpoint, args.data, args.output, device=args.device, tta=args.tta)
    elif args.command == "predict-cv":
        from iceberg.cv import predict_cv

        result = predict_cv(args.run_dir, args.data, args.output, device=args.device)
    elif args.command == "verify":
        result = verify_run(args.run_dir, args.data)
    else:
        from iceberg.tracking import track, track_cv

        if not args.tracking_uri:
            parser.error("Set MLFLOW_TRACKING_URI or supply --tracking-uri")
        importer = track_cv if args.command == "track-cv" else track
        result = importer(args.run_dir, args.tracking_uri, resume=args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
