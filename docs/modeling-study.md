# Most of the angle gain comes from a population absent at test time

This study uses a conventional competition model to examine evaluation and reproducibility decisions. Its role in the portfolio is engineering validation; the component comparison is not presented as a novel modeling contribution.

**The angle branch's apparent improvement largely answers the wrong population question.** With four-flip inference fixed, adding angle metadata lowers all-row out-of-fold log loss from **0.22030 to 0.20479**. But **95.0% of that improvement comes from 133 missing-angle records, all ships**, while the competition test records have known angles. The practical decision is to judge the branch primarily on the known-angle cohort and retain the image-only model as a credible simpler comparison.

The path from score to conclusion is a decomposition. Log loss averages over records, so the all-row change can be separated into the record-weighted changes for missing and known angles. Missing-angle loss falls from **0.20716 to 0.02946**; known-angle loss moves only from **0.22148 to 0.22064**. That explains where the aggregate improvement occurs. It does not prove that the network learned a missingness shortcut: the branch adds both angle and missingness inputs, changes the architecture, and adds parameters. The data are consistent with a useful training-population association that need not benefit records with known angles.

A second check asks whether the small known-angle gain survives training variation. It does not have a consistent sign: the matched seed effects are **+0.00124** and **−0.00278**, and two of five ensemble folds get worse. Without flip averaging, the angle branch also worsens known-angle ensemble loss. Thus neither the large all-row gain nor its consistency across seeds establishes that angle metadata improves the population most relevant to the test set. Matching metadata availability is still only one aspect of matching that population.

The same study offers a more defensible improvement to retain: **four-flip inference lowers known-angle ensemble loss for both architectures**, by **0.00229** for image only and **0.00479** for image plus angle. Each seed improves when its own saved weights are reused, reducing the ambiguity from refitting a different architecture. The effect is not universal across folds, and it costs four forward passes per model.

The resulting research decision concerns the claim and comparison, not a retrospectively selected benchmark. Keep `configs/metadata.json` as the frozen reference, report the known-angle and all-row results together, and expose the matched image-only model with flips alongside it. No default or leaderboard submission was changed. Separating the angle value from its missingness indicator would be a useful next hypothesis; this study does not isolate those effects.

The [executed notebook](../notebooks/02-controlled-ablations.ipynb) visualizes the four conditions, both seeds and every paired fold effect. [Aggregate evidence](modeling-study.json) retains their metrics and provenance.

## Two controlled questions

1. Does adding the optional angle branch help on held-out incidence-angle groups? Compare image-only and metadata models with four-flip inference fixed.
2. Does flip averaging help the same trained model? Compare no augmentation against four-flip inference separately for each set of saved weights.

Both trained conditions use the same 1,604 labeled records, five grouped folds, split seed 25, training seeds 1017 and 2027, 30 fixed epochs, optimizer, learning-rate schedule, training augmentation and batch size. The final epoch is evaluated; held-out losses do not select a checkpoint. The study retains **20 fitted models**: ten per architecture. The no-flip conditions require only inference from those checkpoints.

Both producer manifests identify revision `88683b8371f2fec2f2836aede2533f15c1eb94f9` and the same package-source fingerprint. Input and fold-assignment hashes match. Only configuration `name` and `use_angle` differ. Each fit used four CPU threads; metadata CV ran two workers concurrently and image-only CV ran one. Elapsed training times are not compared, and deterministic algorithms were not forced.

Four-flip inference averages probabilities for the original image, horizontal flip, vertical flip and both flips. Both radar channels undergo the same transform. The two-seed OOF ensemble then averages the two held-out probabilities for each record; its loss is not the arithmetic mean of the two seed losses.

## Four retained conditions

Lower log loss is better. Scores below use mean probabilities across the two predeclared seeds.

| Condition | Known-angle loss: 1,471 rows | All-row loss: 1,604 rows | Missing-angle loss: 133 ships |
|---|---:|---:|---:|
| Image only, no flips | 0.22377 | 0.22287 | 0.21298 |
| Image only, four flips | 0.22148 | 0.22030 | 0.20716 |
| Image + angle, no flips | 0.22544 | 0.20934 | 0.03124 |
| Image + angle, four flips | 0.22064 | 0.20479 | 0.02946 |

All 133 missing-angle records are ships, whereas the competition test metadata contains known angles. Adding the branch reduces missing-angle loss from 0.20716 to 0.02946. Weighting the stratum changes by their record counts attributes **95.0% of the observed all-row loss improvement** to these missing-angle rows. This is an arithmetic decomposition, not proof of a causal mechanism or evidence that the same benefit transfers to the test population.

Without flip averaging, adding the branch actually increases known-angle ensemble loss from 0.22377 to 0.22544. The branch's apparent value therefore also depends on the inference procedure.

## Paired seed and fold effects

Each delta is **after minus before**; negative favors the added component. Seed numbers identify matched recipes, not identical initialization across different architectures.

| Controlled addition | Known-angle ensemble delta | Seed 1017 known-angle delta | Seed 2027 known-angle delta | All-row ensemble delta |
|---|---:|---:|---:|---:|
| Angle branch, flips fixed | −0.00084 | +0.00124 | −0.00278 | −0.01550 |
| Four flips, image-only weights fixed | −0.00229 | −0.00768 | −0.01419 | −0.00258 |
| Four flips, metadata weights fixed | −0.00479 | −0.01413 | −0.00725 | −0.00454 |

The all-row seed deltas also remain in the aggregate receipt. For the angle branch they are −0.01358 and −0.01771; this consistency across all rows does not resolve the conflicting known-angle effects.

The angle comparison below retains the weak folds rather than selecting a favorable subset:

| Held-out fold | Seed 1017 known-angle delta | Seed 2027 known-angle delta | Two-seed ensemble delta |
|---|---:|---:|---:|
| 0 | +0.00084 | −0.00998 | −0.00540 |
| 1 | −0.01563 | −0.02237 | −0.01876 |
| 2 | +0.00682 | +0.01482 | +0.01106 |
| 3 | −0.00360 | −0.00260 | −0.00434 |
| 4 | +0.01794 | +0.00679 | +0.01363 |

Flip averaging improves nine of ten seed/fold comparisons in each architecture. After averaging seeds, it improves three of five image-only folds and all five metadata folds. These counts describe this fixed experiment; correlated folds and two seeds do not support confidence intervals or significance claims. Seed spread, fold difficulty and the effect of averaging predictions are distinct quantities.

## Boundaries of the finding

- The angle branch includes standardized angle and missingness, adding 1,072 parameters: **310,497 → 311,569**. Architecture, parameter count and initialization all change. This tests the branch as implemented, not the isolated causal value of angle information.
- Exact incidence angle is an acquisition proxy, not a verified scene identifier. Known-angle validation matches test metadata availability, not necessarily the entire test distribution.
- The same five folds were used throughout; split-seed sensitivity was not measured. Model development preceded these diagnostics, so this is not an untouched final evaluation.
- The nominal best aggregate condition is not a universal winner. The angle gain reverses in one seed and two ensemble folds, while the simpler model is close on the primary cohort.
- Four-flip inference requires four forward passes per model instead of one; this study measures predictive effects, not an optimized latency benchmark.

## Reproduce and inspect

The notebook reruns from the public aggregate JSON without private data or training. With the repository environment installed:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m jupyter nbconvert \
  --execute --to notebook --inplace notebooks/02-controlled-ablations.ipynb
```

`configs/image-only-flips.json` exposes the equally budgeted image-only comparison. Its descriptive name differs from the producer label (`modern-image-only-controlled-ablation`); every scientific parameter matches the recorded recipe after `--device cpu` resolves the device. `configs/metadata.json` remains the frozen reference; CLI defaults are unchanged. Do not substitute `configs/baseline.json` unchanged: its default inference setting is `none`.

```bash
.venv/bin/iceberg cv --data-dir "$ICEBERG_DATA_DIR" \
  --config configs/metadata.json --run-dir .local/ablation-metadata \
  --strategy grouped --folds 5 --split-seed 25 --seeds 1017 2027 \
  --device cpu --workers 2
.venv/bin/iceberg cv --data-dir "$ICEBERG_DATA_DIR" \
  --config configs/image-only-flips.json --run-dir .local/ablation-image \
  --strategy grouped --folds 5 --split-seed 25 --seeds 1017 2027 \
  --device cpu --workers 1
```

Use fresh output directories. The input snapshot and producer fingerprints in the receipt identify the reported run; a later rerun is a separate experiment.

Given completed runs:

```bash
.venv/bin/python scripts/analyze_ablation.py \
  --metadata-run .local/ablation-metadata --image-run .local/ablation-image \
  --data "$ICEBERG_DATA_DIR/train.json" --output .local/ablation-comparison
.venv/bin/python scripts/verify_ablation.py \
  --metadata-run .local/ablation-metadata --image-run .local/ablation-image \
  --data "$ICEBERG_DATA_DIR/train.json" --comparison .local/ablation-comparison \
  --output .local/ablation-comparison/verification.json
```

The analyzer verifies completed parent/child bundles, matched source/data/splits and the controlled configuration difference before inference. The independent verifier joins retained predictions to raw labels and fold IDs, then computes binary cross-entropy directly with NumPy. All **76 metric reconciliations** passed, with maximum absolute error **8.33 × 10⁻¹⁷**. Raw predictions, record identifiers and weights remain outside Git.
