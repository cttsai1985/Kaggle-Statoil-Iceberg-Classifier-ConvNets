# Prepare an offline Kaggle notebook using the same package

The educational walkthrough lives in `notebooks/`. The thin `kaggle/rerun.ipynb` entry point installs a code wheel from an attached dataset, runs the bounded model recipe and writes test predictions plus a run bundle.

## Prepare code assets locally

```bash
.venv/bin/python tools/export_kaggle.py
```

The ignored `kaggle-bundle/code-dataset` contains the wheel, checksum and license. The `kernel` folder contains the notebook and metadata. No competition data, trained weights or credentials are staged. Change owner/slug metadata when using another account.

## Explicit account operations

Inspect staged files first. The following commands write to Kaggle; preparation itself never uploads.

```bash
# First code dataset version, private by default:
.venv/bin/kaggle datasets create -p kaggle-bundle/code-dataset
# Use datasets version for later versions of an existing dataset.
.venv/bin/kaggle kernels push -p kaggle-bundle/kernel
.venv/bin/kaggle kernels status cttsai1985/iceberg-modern-method-rerun
.venv/bin/kaggle kernels output cttsai1985/iceberg-modern-method-rerun \
  -p .local/kaggle-output
```

Push normally runs the notebook. Use `--no-run` to save without executing. The metadata defaults to private, CPU and no internet, with the official competition and code dataset attached.

The entry point expects extracted `train.json` and `test.json` in the competition mount. If Kaggle supplies compressed archives, add a compatible offline extraction step/dependencies before running. Its installed scientific packages must satisfy the wheel requirements; the full hosted image is not assumed identical to the Docker lock.

The notebook writes `submission.csv` and `run-bundle.zip`. Before importing, retrieve the exact notebook version, record its version/wheel identity, safely extract into a fresh local directory and verify manifest/artifact hashes. Remote-version receipts and safe bundle extraction are not automated yet.

Remote Kaggle execution is unverified. Leaderboard submission is separate and depends on this closed competition's current eligibility. No Kaggle publication, execution or submission was performed in this PR.
