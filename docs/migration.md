# Preserve the method and make execution inspectable

The original solution combines two radar polarizations and incidence angle through a multi-input CNN. The modern code retains both convolution branches and keeps the original scripts unchanged at reference commit d7f9f46145f067a82d3ddf3a9e6ca134bd098083.

| Decision | Preserved behavior |
|---|---|
| Radar input | Two 75×75 bands, rescaled with `x / 100 + 0.5` |
| Metadata | Missing string `na` becomes `-1` |
| Split | StratifiedShuffleSplit, seed 25, holdout fraction 0.159 |
| Image branches | Four pooled three-convolution blocks plus a six-convolution full-resolution branch |
| Activations | ELU blocks, binary probability output |
| Normalization | Momentum-zero running statistics, epsilon 0.001, population variance |
| Augmentation | Shared channel geometry, original probabilities, centered nearest-fill affine transforms |
| Optimizer intent | Adam at 0.0005; inverse-step learning-rate decay 0.002 |
| Checkpoint selection | Lowest held-out log loss |

## Changes and rationale

- PyTorch replaces Keras 2 APIs. Tensor layout changes from NHWC to NCHW. BCEWithLogitsLoss combines the original sigmoid and BCE for training; prediction applies sigmoid.
- Explicit normalization makes the Keras running-variance policy inspectable and supports a final minibatch with one angle value. Native PyTorch batch norm has different running-variance updates.
- Python, NumPy and PyTorch are seeded. The original generator mixed NumPy and unseeded Python randomness. Random sequences and initialization still differ across frameworks.
- Download, validation, training, evaluation and prediction have separate entry points. Evaluation verifies the training-file hash and uses saved holdout indices. The legacy inference script's training-data score is not presented as held-out validation.
- The original fixed zoom range `(0.1, 0.1)` is preserved. Changing it to a common ±10% interval would be a new experiment.
- The bounded profile retains original model width and holdout but trains on 128 records for one epoch and one pass. The reference configuration requests all training rows, three epochs and eight passes. Its full training recipe remains unverified.
- The original `ceil(8*n/batch)` generator step count can stop part-way through its eighth traversal. The maintained eight-pass setting completes eight traversals, so optimizer-step identity is not claimed.
- The old callbacks have patience 40/100 and cannot trigger during a fresh three-epoch run; they are not ported. The optimizer's old `decay` is mapped to learning-rate decay, not L2 weight decay.

## What the evidence establishes

The maintained implementation runs on real data, reloads checkpoints and produces complete test outputs. Numerical Keras/PyTorch equivalence, historical leaderboard parity, CUDA and full-reference training remain unverified.

The walkthrough explains data modalities, missing metadata, validation, architecture, error distributions and saved artifacts. One-factor experiments are suggestions for learning; no improvement claim is inferred from one short run.

Source: [original scripts](https://github.com/cttsai1985/Kaggle-Statoil-Iceberg-Classifier-ConvNets/tree/d7f9f46145f067a82d3ddf3a9e6ca134bd098083/scripts).

