# Kaggle notebook execution

The educational walkthrough lives in `notebooks/`. The thin `notebooks/kaggle/rerun.ipynb` entry point installs a code wheel from an attached dataset, runs the complete metadata-model recipe and writes test predictions plus a run bundle.

## Prepare code assets locally

```bash
.venv/bin/python scripts/export_kaggle.py
```

The ignored `.local/kaggle-bundle/code-dataset` contains the wheel, checksum and license. The `kernel` folder contains the notebook and metadata. No competition data, trained weights or credentials are staged. Change owner/slug metadata when using another account.

## Explicit account operations

Inspect staged files first. The following commands write to Kaggle; preparation itself never uploads.

```bash
# First code dataset version, private by default:
.venv/bin/kaggle datasets create --keep-tabular -p .local/kaggle-bundle/code-dataset
# Wait for the dataset status to become ready before pushing the kernel.
.venv/bin/kaggle datasets status cttsai/kaggle-iceberg-code
# Use datasets version for later versions of an existing dataset.
.venv/bin/kaggle kernels push -p .local/kaggle-bundle/kernel
.venv/bin/kaggle kernels status cttsai/kaggle-iceberg-solution-mlops
.venv/bin/kaggle kernels output cttsai/kaggle-iceberg-solution-mlops \
  -p .local/kaggle-output
```

After pushing, pull the kernel metadata and verify both the code dataset and official competition remain attached. A dataset still processing can cause an unresolved attachment to be dropped.

Push normally runs the notebook. Use `--no-run` to save without executing. The metadata defaults to private, CPU and no internet, with the official competition and code dataset attached.

The entry point finds extracted payloads or extracts the official 7z archives using the runtime's `py7zr` or 7-Zip executable. It verifies the attached wheel/configuration fingerprints and installed dependency versions before training. Hosted scientific packages are recorded independently from the local Docker lock.

The notebook writes `submission.csv` and `run-bundle.zip`. Before importing, retrieve the exact notebook version, record its version/wheel identity, safely extract into a fresh local directory and verify manifest/artifact hashes. Remote-version receipts and safe bundle extraction are not automated yet.

Remote Kaggle execution is unverified. Leaderboard submission is separate and depends on this closed competition's current eligibility. No Kaggle publication, execution or submission was performed in this PR.
