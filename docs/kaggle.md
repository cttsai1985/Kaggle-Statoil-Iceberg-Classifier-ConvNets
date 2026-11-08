# Kaggle notebook execution

The educational walkthrough lives in `notebooks/`. The thin `notebooks/kaggle/rerun.ipynb` entry point installs a code wheel from an attached dataset, runs the complete metadata-model recipe and writes test predictions plus a run bundle.

## Prepare code assets locally

```bash
.venv/bin/python scripts/export_kaggle.py
```

The ignored `.local/kaggle-bundle/code-dataset` contains the wheel, checksum and license. The `kernel` folder contains the notebook and metadata. No competition data, trained weights or credentials are staged. Use `--owner YOUR_KAGGLE_USERNAME` for another account. The exporter keeps the title and slug aligned: `Kaggle Iceberg Solution MLOps` maps to `kaggle-iceberg-solution-mlops`.

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
.venv/bin/kaggle kernels output cttsai/kaggle-iceberg-solution-mlops/1 \
  --file-pattern '(run-bundle\.zip|submission\.csv|submission\.receipt\.json)$' \
  -p .local/kaggle-output-v1
```

After pushing, pull the kernel metadata and verify both the code dataset and official competition remain attached. A dataset still processing can cause an unresolved attachment to be dropped.

Push normally runs the notebook. Use `--no-run` to save without executing. The metadata defaults to private, CPU and no internet, with the official competition and code dataset attached.

The entry point finds extracted payloads or extracts the official 7z archives using the runtime's `py7zr` or 7-Zip executable. It verifies the attached wheel/configuration fingerprints and installed dependency versions before training. Hosted scientific packages are recorded independently from the local Docker lock.

The notebook writes `submission.csv` and `run-bundle.zip`. Retrieve an explicit version as above, rather than whichever version is latest. Extract into a fresh directory only after checking that the archive has unique, flat filenames, no symbolic links, and bounded member/total sizes. Then run `iceberg verify` against the original training data, check the submission receipt against the test data and output hashes, and confirm every test ID appears exactly once in its original order. Archive extraction and the remote-version receipt are currently manual steps.

## Verified hosted execution

The private [Kaggle notebook](https://www.kaggle.com/code/cttsai/kaggle-iceberg-solution-mlops) completed version **1** on November 8, 2026, using code dataset `cttsai/kaggle-iceberg-code/1` built from clean commit `0e042cb8c64988923322ceddc70852f5ede9b857`. This was the complete 30-epoch metadata-model holdout recipe. The separate grouped cross-validation evaluation remains the primary model-quality evidence; this hosted run verifies portability.

| Check | Observed result |
| --- | --- |
| Hosted runtime | Python 3.13.18, PyTorch 2.11.0+cpu, NumPy 2.1.3, SciPy 1.16.3, scikit-learn 1.6.1 |
| Holdout | 256 rows; log loss 0.205951803; known-angle log loss 0.216443679 |
| Prediction output | All 8,424 test IDs, complete and in original order |
| Input and artifact identity | Official train/test hashes, code wheel/configuration/source hashes, checkpoint and output receipts verified |
| Local CPU reload | All stored evaluation metrics reproduced exactly; allowed absolute tolerance was 1e-5 |
| Shared MLflow round trip | Imported into `kaggle/iceberg`; repeat import reused the same run; downloaded checkpoint SHA-256 matched |

The hosted and original local metadata runs used identical input hashes, resolved configuration and saved split IDs. Their source revisions and scientific-library versions differed, so the independently trained scores are not a claim of bitwise training reproducibility. Reloading the same hosted checkpoint is the portability check above.

The stable Kaggle notebook ID is `143481450`. Version 1 originally ran under `cttsai/iceberg-solution-reproducible-pytorch-and-mlops`. Version **2** corrected the title and canonical slug with `kernels push --no-run`, retaining that notebook ID and the completed version-1 outputs. Version 2 is a metadata save, not another executed benchmark. The code dataset remains version 1. For future title changes, pull metadata first and retain its `id_no` in the staging directory; do not place account-specific numeric IDs in the reusable repository template.

Local verification evidence is retained in ignored `.local/hosted-verification.json`, with the downloaded producer manifest unchanged. The independently downloaded code wheel SHA-256 is `ed386693812658d6ed8241152bdb92395c2e6c0bf5779b0987f8378cfc87724f`.

The notebook and code dataset remain private. Publishing a notebook and submitting its predictions to the leaderboard are separate actions; this hosted verification did not make a competition submission.
