# Zoom severity: predeclared matched comparison

Question: does the historical extreme zoom affect this modern model's held-out
log loss when its 5% application probability and the remaining recipe are fixed?
This protocol is committed before fitting. It is not an attempt to infer the
original author's intention or reproduce historical Keras weights.

Three conditions use no zoom (`none`), Keras affine scale bounds `[0.9, 1.1]`
(`mild_5pct`), or the historical fixed `[0.1, 0.1]` (`extreme_5pct`). All consume
identical random draws and have the same 5% trigger schedule. The no-zoom arm uses
the identity at those triggers. Keras scale is an output-to-input affine map:
`0.1` produces approximately 10x visual magnification, not +/-10% zoom.

The existing default `current` is unchanged. For the three study arms, remove the
scale coupled to the modern rotation/shift block, retain its other transforms,
and insert the isolated Keras-style zoom (nearest interpolation/fill and historical
centering). Thus the comparison changes only zoom severity across study arms;
none of them is advertised as a numerical replay of the normal pipeline or all
historical augmentations. Two equal mild bounds would not be a randomized range.

Use all 1,604 labels, five fixed grouped folds (exact-angle acquisition proxy,
split seed 25), seeds 1017 and 2027, metadata CNN and four-flip inference from the
frozen metadata recipe. Fit preprocessing on each training fold only. Fix **30
natural epochs** per model and evaluate the final epoch; no validation-selected
checkpoint, threshold/weight tuning, filtered score or new submission. Total:
**30 models**. CPU only, four workers with two threads each (eight declared training
threads); no GPU, shared-service change or unbounded parameter search.

Primary endpoint: two-seed ensemble OOF log loss on the 1,471 known-angle records,
matching test metadata availability. Also retain all-row and missing-angle scores,
every seed/fold result, signed paired deltas, high-loss concentration, input/split/
source/config/checkpoint hashes and independent reconstruction of every checkpoint.
The 133 missing-angle records are all ships. Shared folds and two seeds do not
supply independent replications or confidence intervals. Existing development
used this dataset, so this is a retrospective diagnosis, not a new final test.

Interpretation is fixed before results: mixed fold/seed signs indicate uncertainty;
consistent lower loss supports that policy only for this controlled setting. A
lack of measured harm is not evidence of equivalence. Do not change defaults or
claim a leaderboard improvement from OOF findings. Original intent stays unknown
unless historical source evidence states it.

Run from a clean committed checkout with authorized inputs and a fresh output:

```bash
.venv/bin/python scripts/study_zoom.py --data "$ICEBERG_DATA_DIR/train.json" \
  --output .local/zoom-study-2026-11-10
```

Public outputs are aggregate Markdown/JSON and an executed notebook; raw radar
images, IDs, predictions and checkpoints stay in ignored local artifacts.
