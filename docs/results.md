# The initial evidence establishes a bounded rerun

The 2017 README reports a score around 0.194–0.205. That remains an author-reported competition result. It was not reproduced or used as this PR's acceptance threshold.

| Observed item | Scope |
|---|---|
| Data | Official Kaggle train/test payloads; hashes retained |
| Training | 128 real records, one epoch/pass, original model width |
| Validation | Original seed-25 stratified holdout, 256 records |
| Holdout log loss | 0.8497995872496394 in the first bounded CPU run |
| Train/evaluate time | 10.279 seconds on the recorded four-thread host |
| Reload | Independent local and Docker evaluation reproduced the saved metric |
| Test prediction | All 8,424 rows; IDs/order matched the official sample |
| Walkthrough | Executed with real data and full-test inference |

The short-budget result has weak predictive performance. It establishes execution and traceable artifacts, not a competitive model or historical reproduction. Training budget, held-out labels and leaderboard context differ from the old reported score.

[The validation receipt](validation.json) records checks and fingerprints without publishing competition records or weights. Full reference training, numerical Keras/PyTorch equality, CUDA, connected MLflow/SeaweedFS import and remote Kaggle execution remain unverified.

