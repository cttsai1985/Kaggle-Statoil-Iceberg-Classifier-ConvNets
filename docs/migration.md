# Model and training changes

The 2017 solution combined two radar bands and incidence angle through two convolutional branches. The maintained pipeline uses a compact PyTorch CNN, current training APIs and explicit experiment artifacts. The historical competition score is the benchmark; numerical emulation of Keras 2 is not an objective.

## Architecture and preprocessing

| Component | Historical implementation | Maintained implementation |
|---|---|---|
| Image input | Two 75×75 bands, fixed `x / 100 + 0.5` scaling | Two native-resolution dB bands, channel statistics fitted on training rows |
| Image encoder | Pooled branch plus six full-resolution convolutions | Four progressively downsampled CNN stages, global average and maximum pooling |
| Normalization | Keras batch normalization with momentum zero | Native BatchNorm2d in the encoder, LayerNorm in the dense head |
| Metadata | Missing angle replaced with numeric `-1` | Optional branch with training-median imputation, scaling and missingness flag |
| Spatial augmentation | Paired flips/rotations/shifts, occasional fixed `0.1` zoom | Paired flips/rotations and mild affine changes |
| Optimization | Historical Adam arguments and repeated generator traversals | Standard AdamW and cosine scheduling over natural epochs |
| Evaluation | Original stratified holdout and competition submission | Primary five-fold grouped CV with two training seeds and fixed final epochs; separate fixed holdout and competition scores for comparison |

Progressive downsampling reduces the cost of full-data experiments. Modern normalization removes the dense head's dependence on the last training minibatch. The new preprocessing retains relative brightness between observations and fits no statistics on held-out or test rows.

The extreme historical zoom is replaced with a mild range. RGB color transforms and swapping the two polarizations are not used. Metadata is a controlled experiment rather than an unconditional feature: acquisition geometry can expose grouping structure, and missingness is not a physical angle.

## Validation and interpretation

The primary [CV protocol](cv.md) audits all folds and both training seeds, including class balance, group separation, OOF coverage and known-angle performance. Fold and seed dispersion are descriptive diagnostics; no uncertainty guarantee follows from two seeds.

The fixed comparison split uses stratification, seed 25 and a 0.159 holdout fraction: 1,348 training records and 256 validation records. Checkpoint selection uses holdout log loss. Reported leaderboard loss is separate; matching a local holdout number does not establish reproduction of a competition score.

Preprocessing is saved with each checkpoint. Evaluation verifies the input identity and reuses its stored holdout. Test predictions use the same saved model and inference configuration as validation, retaining all official IDs in order.

The historical source does not fully specify all random states or preserve trained checkpoints. Modern training changes are explicit, and benchmark comparisons use observed scores rather than claims of identical historical weights.

## Historical source

The original files remain under [`legacy/`](../legacy/), including their relative directory structure. Their provenance is [commit d7f9f46](https://github.com/cttsai1985/kaggle-statoil-iceberg/tree/d7f9f46145f067a82d3ddf3a9e6ca134bd098083/scripts).

[Benchmark results](results.md) · [Experiment runbook](reproduction.md)
