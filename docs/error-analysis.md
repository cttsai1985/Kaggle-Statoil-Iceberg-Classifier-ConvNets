# A small high-loss cohort persists across both training seeds

**The highest-loss 161 of 1,604 grouped OOF observations contribute 73.02% of total log loss. Of these, 160 have known incidence angles, and 134 also appear in each individual seed's hard cohort.** The retained errors therefore justify investigating difficult acquisition conditions and probability behavior on known-angle targets, alongside the already measured image/metadata comparison.

The cohort is fixed as the largest **ceiling(10% × rows)** per-row binary cross-entropies of the recorded two-seed metadata ensemble, with retained-row order breaking ties. “Hard” means high probabilistic loss; some selected observations can still be correctly classified at a 0.5 cutoff. This is a retrospective diagnostic, not a new test population or a model improvement.

## Which observations carry the loss

| Cohort | Population | High-loss records | High-loss rate | Mean log loss |
|---|---:|---:|---:|---:|
| Known angle below 35° | 259 | 12 | 4.63% | 0.11209 |
| Known angle 35° to below 40° | 572 | 62 | 10.84% | 0.21400 |
| Known angle at least 40° | 640 | 86 | 13.44% | 0.27052 |
| Missing angle | 133 | 1 | 0.75% | 0.02946 |

The whole high-loss cohort contains **82 ships and 79 icebergs**. Both classes therefore need attention; the pattern is not a collapse on only one label. The angle buckets are fixed descriptive cuts, not discovered physical regimes. Class mix, image appearance and acquisition conditions can differ across them; the association does not identify a causal angle effect.

High-loss records appear in every fold: **32, 31, 29, 37 and 32**. The weakest fold has greater mean loss without containing most of the difficult observations. Its records should not be discarded to improve a headline score.

## Which difficulties survive another seed

Each individual seed contributes a separately selected 161-record hard cohort. They share **134 records**, with union 188 and **Jaccard overlap 0.713**; those same 134 records are also in the ensemble cohort. This identifies a repeatable review set under the recorded seed variation, rather than proving that the labels are wrong or that the errors are irreducible. Two seeds share one fold plan, so this overlap is not a confidence interval.

The independently recomputed full-population log loss is **0.204792**, matching the retained result. Known-angle observations account for **98.81% of the total loss**. This complements the [controlled modeling study](modeling-study.md): most of the angle branch's aggregate gain came from missing-angle ships, while four-flip inference improved known-angle results in both tested architectures. These diagnostics do not choose a replacement model or change the frozen recipe.

## The next controlled question

First compare the **same retained predictions** from image-only/metadata and no-flip/four-flip conditions within the predeclared angle and class strata. A benefit concentrated in one acquisition range would motivate a different follow-up from a benefit shared across classes and ranges. Retain all strata and all rows; do not select a new score population from the observations that look favorable.

A private review can then examine whether the persistent hard cases differ in radar intensity/contrast or ambiguous appearance, while keeping HH/HV alignment intact. Such inspection would generate hypotheses about representation or augmentation. It cannot justify relabelling a difficult example merely because both models disagree with it.

For new data, acquire actual scene/time identifiers and evaluate held-out acquisitions. Exact incidence angle remains an **acquisition proxy**, not proof that entire satellite scenes are independent. The current analysis cannot establish how an unseen sensor, region or period would behave.

## Reproduce the diagnostic

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/analyze_hard_cases.py \
  --run-dir .local/runs/cv-a02b6d40 --data "$ICEBERG_DATA_DIR/train.json" \
  --output docs/error-analysis.json
```

The analyzer streams raw labels/angles without retaining radar pixel arrays, verifies input/parent/child/checkpoint hashes, checks exact ID coverage and grouped assignments, and reconstructs each ensemble probability from the two seed OOF values. It computes binary cross-entropy independently before comparing with the retained pooled metric. [Aggregate evidence](error-analysis.json) contains no IDs, raw images or per-record labels. No fitting, inference, filtering, relabelling or promotion occurs.
