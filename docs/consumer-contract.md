# Iceberg: the MLOps consumer contract

The solution owns the modeling question, validation method and acceptance criteria. It consumes shared execution, tracking and storage services. Its scientific claims do not become stronger merely because a run appears in MLflow.

## From scientific evidence to an executable job

| Boundary | This solution supplies | What it expects from shared services |
|---|---|---|
| Inputs | Authorized data, schema/ID checks and exact input hashes | Read-only input access and explicit identity on execution |
| Experiment | Fixed configuration, folds, seeds, preprocessing and budget | An immutable environment/image reference and recorded task state |
| Evaluation | OOF/holdout analyzer, retained weak folds, variance and ablation evidence | Durable results and logs for both favorable and unfavorable experiments |
| Delivery | Verified checkpoint and ordered batch outputs | Artifact transfer with checksummed readback; explicit failure/recovery |
| Troubleshooting | A portable local run bundle and project-specific validation errors | Stage/flow correlation, owner-visible status and safe retry semantics |

The canonical tracking experiment is `kaggle/iceberg`. The platform provisions it; this client does not silently create a different namespace. The run retains its own logical identity and source/input/model fingerprints, including when it originates in an offline Kaggle notebook.

## Failure behavior matters to research

A transient tracker outage must not require retraining an expensive completed experiment. `track --resume` recovers the same logical/remote run only after the operator confirms the earlier importer has stopped. A conflicting manifest or duplicate logical identity fails closed. Artifact delivery is marked finished only after byte-hash readback. [Project-specific commands and recovery details](mlops.md).

A poor model remains an experiment worth recording. A delivery gate may reject a candidate without hiding its metrics. Acceptance thresholds must be declared for the intended use and evaluated on the right population; a convenient competition holdout threshold is not production approval.

## Evidence and limits

The [modeling study](modeling-study.md) tests component contributions and descriptive variability. The [shared batch case study](https://github.com/cttsai1985/mlops-platform/blob/main/docs/batch-operations.md) demonstrates an Iceberg saved-model workflow with real inputs, task logs, delivery gating, failure and recovery. It does not imply every historical experiment or every solution ran under that scheduler.

Current CI and recovery tests exercise the client contracts. The [capability assessment](https://github.com/cttsai1985/mlops-platform/blob/main/docs/mlops-assessment.md) distinguishes these controls from live service monitoring, automatic retraining, model promotion and disaster recovery. This repository makes no claim to operate those production capabilities.
