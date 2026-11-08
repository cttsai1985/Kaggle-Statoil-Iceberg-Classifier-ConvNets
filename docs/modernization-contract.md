# A common portfolio format with room for each problem

The portfolio should let a reader understand the modeling decision, rerun a declared experiment and inspect its evidence. Competition rules and original author instructions take precedence over template convenience.

## Common outcomes

| Reader question | Repository evidence |
|---|---|
| What problem does this solve? | Short README with task, inputs, approach and measured scope |
| Why this method? | Markdown rationale and a narrated data-science notebook |
| What changed from the original? | Preserved legacy source and an explicit migration note |
| Can I run it? | Locked Python environment, CLI, Docker Lab and dev container |
| Can I trust the reported result? | Declared split, resolved configuration, input/source hashes and checkpoint reload |
| How does it fit MLOps? | Portable run artifacts and optional clients for existing services |
| Can it run on Kaggle? | Thin notebook entry, code packaging and a separately recorded hosted-runtime check |

These are outcome requirements. A text-labeling model, a radar CNN and a relational credit model should retain their own features, validation logic, metrics and hardware needs. They do not need identical model classes or notebook counts.

## Execution profiles keep costs and claims explicit

1. **Fixture:** generated data tests software behavior without competition access.
2. **Bounded real-data run:** a small declared training budget verifies the actual input-to-output path. Its metric is not a leaderboard claim.
3. **Reference recipe:** the documented larger experiment, promoted to verified only after execution and review.
4. **Hosted Kaggle run:** a separately validated runtime with attached inputs, internet constraints and recorded notebook/code versions.

Local notebooks and CLI runs use the same project package. Hosted notebooks may use a different scientific environment and may lack access to the shared network. They retain portable evidence; a later explicit import connects that evidence to tracking. Service availability should not determine whether training artifacts survive.

## Infrastructure boundaries

Competition repositories own source, tests, notebooks, runtime images, execution configuration and small client adapters. Shared infrastructure owns service deployment, upgrades, storage administration, credentials, backups and access control.

MLflow is the first tracking adapter in this implementation. An existing SeaweedFS client mount can expose shared files read-only to Lab. Prefect orchestration and W&B mirroring require concrete integration work and their own verification before being advertised as working capabilities. No service upgrade is required merely to modernize a model repository.

## Rollout and acceptance

| Repository | First modernization focus | Status in this change |
|---|---|---|
| Statoil Iceberg | Structurally faithful CNN port, original holdout, complete test inference | Implemented and bounded CPU run verified |
| Home Credit Default Risk | Inspect relational feature pipeline and leakage controls before selecting a current baseline | Planned; no changes here |
| Google QUEST QA | Inspect target definitions, grouping and legacy text-model dependencies before selecting a current runtime | Planned; no changes here |

For each follow-up, inspect the actual repository and current competition delivery requirements before copying this scaffold. Document differences and validate the smallest meaningful end-to-end experiment first. Repository renaming is a separate task.

The immediate Iceberg follow-ups are full reference training, hosted Kaggle validation, and a verified import into the existing shared tracking/storage deployment. See [observed results](results.md) for the current evidence boundary.
