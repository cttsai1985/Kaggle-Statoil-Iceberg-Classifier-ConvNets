"""Export an already evaluated Iceberg checkpoint as an MLflow Model."""

import argparse
import json
from pathlib import Path

from iceberg.serving import export_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--threads", type=int, default=2, choices=[1, 2, 4])
parser.add_argument("--max-batch-rows", type=int, default=64)
args = parser.parse_args()
print(
    json.dumps(
        export_model(
            args.run_dir, args.output, threads=args.threads, max_batch_rows=args.max_batch_rows
        ),
        indent=2,
    )
)
