# ML solution portfolio standard

Version: **1.0** · Established: **2026-11-09**

This is the accumulated acceptance standard for the solution portfolio and its
shared infrastructure. New branding, prose, tools or deployment work must preserve
these requirements. The solution remains the unit of presentation: problem,
modeling, evidence, usable predictions and an explicit operating handoff.

P0 requirements must pass, or have a reviewed and explicit applicability decision,
before a portfolio correction is called complete. P1 requirements remain accepted
work with visible status; they cannot disappear or be described as completed
without evidence. A canary proves only its recorded scope.

## Stable acceptance requirements

| ID | Priority | Acceptance condition |
|---|---|---|
| R01 | P0 | Maintain the full accumulated requirements across branding, implementation and release changes. Each repository maps requirements to evidence or an explicit open item. |
| R02 | P0 | Lead with a complete ML solution: problem, useful output, measured model decisions, validation, and an executable path from supported inputs to predictions. |
| R03 | P0 | Keep score and ranking context visibly paired in the primary README results display and GitHub description; surface historical and current results without selective omission. |
| R04 | P0 | Every reported score identifies its metric, direction, model/run, evaluation population and protocol. Leaderboard scores additionally carry matched reference position/N, top share, ties and snapshot. |
| R05 | P0 | Distinguish official awarded rank from score-equivalent reference position, public from private, historical account results from source-reproduced models, and OOF from leaderboard evaluation. |
| R06 | P0 | Show substantive ML insights: hypothesis → controlled comparison → effect/variation → practical decision → remaining question. Modern tooling alone is not model improvement. |
| R07 | P0 | Show project-specific error analysis and hard cases, including the path from input characteristics through representation/prediction to failure. State what is already measured and what still needs analysis. |
| R08 | P0 | Preserve trustworthy evaluation: exact splits/IDs, training-only transformations where implemented, grouped leakage controls, all-row OOF coverage, fold/seed sensitivity, independent metric checks, and explicit exceptions. |
| R09 | P0 | Do not turn post-hoc hard-case selection into a new unbiased score, delete difficult rows, retune weights/thresholds on reported OOF labels, or infer causal explanations from correlations. |
| R10 | P0 | Preserve a real evaluated model and its preprocessing, tokenizer/features, inference policy, dependencies and provenance; it must predict after reload. |
| R11 | P0 | Test the actual model-to-application interface end to end: native output → exported MLflow package → fresh-workspace HTTP inference → numerical parity, invalid-request rejection and cleanup. |
| R12 | P0 | Declare the actual input boundary and make upstream preprocessing reproducible. A feature-vector endpoint must not be marketed as a raw-data service. |
| R13 | P0 | Preserve local CLI, Docker/dev environment and offline Kaggle workflows around the same maintained code; notebooks explain or invoke that code. |
| R14 | P0 | Use generic shared infrastructure and one-way solution→platform dependency. Clients own names, model packages, environments and evaluation; the platform has no hardcoded client catalog. |
| R15 | P0 | Keep deployment UX lightweight and explicit: protected state, stable target identity, safe reuse, useful status/logs, no heavy host SDK install merely to start infrastructure. |
| R16 | P0 | Preserve existing live services, volumes, ports, credentials, experiments, artifacts and historical evidence through code/repository renames. |
| R17 | P0 | Publish no raw private inputs, request rows, row-level predictions, weights, credentials or diagnostic logs. Public receipts contain inspectable aggregate provenance and outcomes. |
| R18 | P0 | Keep code/lock/model provenance exact; test the intended release source, retain clean/dirty status and hashes, and distinguish old evidence from freshly rerun checks. |
| R19 | P0 | Release all four solutions plus shared infrastructure as explicit canary prereleases with meaningful checks and scope. Canary is not a claim of production traffic splitting. |
| R20 | P0 | Use credible names and solution-first descriptions without assignment-style “MLOps consumer demo” or novelty-disclaimer introductions. Retain attribution and legacy source. |
| R21 | P0 | Explain the business/domain problem from primary sources and distinguish measured model performance from proposed operating benefit or ROI. |
| R22 | P1 | Complete production research/batch and online-serving capabilities proportionately, using MLflow serving where sufficient; do not redefine the target as a demo to avoid work. |
| R23 | P1 | Decide hosting, ownership, service objectives, capacity and recovery targets from real needs; implement appropriate authentication/TLS, observability/alerts, off-host backup/restore and rollout/rollback evidence. |
| R24 | P1 | Preserve batch recovery semantics: immutable inputs, verified completed-output reuse, bounded safe retries, quality gates and understandable failure records. |
| R25 | P1 | Turn retained OOF/holdout evidence into deeper error slices and specific next modeling hypotheses, without unnecessary retraining or a new analytics framework. |
| R26 | P0 | Keep repository URLs, package/export metadata, releases and branch hygiene coherent after approved renames. Delete only verified merged branches after preserving outstanding work. |

## Mandatory result presentation

The README contains a **Results** section immediately after the brief problem and
approach opening, before architecture, reproduction or infrastructure detail. A
public result carries its named model/run, metric/direction, score, reference
position and team denominator, Top %, ties, and dated source. The GitHub description
also retains a clearly scoped score/position/Top % summary. Both historical and
modern results remain visible, including weaker modern results.

Reference position = 1 + strictly better displayed reference scores. Top % =
100 × starting position / unique scored reference teams. Use exact decimal score
comparisons and the correct metric direction. Show a tie interval when needed;
a score absent from the snapshot has an insertion position, not an awarded rank.
Outperformed percentile and Top % are different measures and must not be interchanged.

Public, private, OOF and holdout are separate populations. OOF/holdout leaderboard
rank and Top % are **not applicable**. A private score without a matching private
reference population has **unavailable private rank**, never an invented public-based
percentile. Source checkpoints must not inherit another model's better score.

Every published public score must have a matching comparison or an explicit
unavailable calculation, including a secondary historical blend. Missing values
are not zeros. Preserve exact source receipts, snapshots, counts and scope.

## Modeling and analysis coverage

Each solution retains at least one controlled component comparison and explains
hypothesis, measured effect/variation, practical decision and next validation.
Modeling, representation and input-path insights accompany the score. Outcomes
that do not improve a metric remain visible.

Error analysis uses the actual retained OOF/holdout population and the relevant
metric. Document the mining rule, metric reconciliation, counts, error concentration
or ranking disagreement, and a specific next experiment. A private case index may
support inspection; publish aggregates and reasoning rather than IDs, text, pixels
or individual financial records. A metric need not decompose into row contributions:
label diagnostics such as rank displacement honestly.

A path analysis traces available inputs through preprocessing/representation to
prediction and observed error; it does not require a new explainability framework
or imply causality. Post-hoc hard cases are exploratory. Do not remove difficult
rows, retune on reported OOF outcomes, or report a filtered score as the full score.
Do not manufacture stochastic evidence by changing seeds of a deterministic fit.

## Presentation and implementation flexibility

The standard fixes coverage and evidence, not model families, notebook layouts,
identical folder trees or compulsory deployment features in every solution.
Runnable scripts own reusable implementation; Markdown and executed notebooks
explain its rationale and results. Tool names are supporting implementation detail.
Problem/value claims use primary sources; proposed ROI or operational outcomes
remain hypotheses until measured in the intended workflow.

Solutions own their names, models, environments, experiments and scientific
acceptance. Shared infrastructure owns generic services and their operational
contracts, with no runtime client catalog. Leaderboard/model-quality requirements
are explicitly not applicable to an infrastructure repository; its service
acceptance evidence and open production requirements take their place.

## Change and release control

Each repository maintains `docs/requirements-audit.md`, mapping every requirement
to evidence, an applicability decision or an open item. Before merge, review the
entire matrix, the README Results display and the GitHub description. The local
portfolio check guards source-backed ranking visibility; it does not replace
scientific or editorial review.

Do not rewrite historical model/scoring/canary evidence to fit a new narrative.
New diagnostics keep their own source hashes, scope and run identity. Preserve
attribution, licenses, raw-data boundaries and existing runtime resource identities.
Canary releases retain source, lock, aggregate evidence and checksums; no implicit
production traffic switch or service migration follows from a repo rename.

After merge, delete only branches verified merged into the current default branch.
Preserve default branches, unmerged work and branches used by active worktrees.
Changes to this standard require a version/change-log update and a complete
compatibility review; a wording or branding request cannot silently weaken it.

## Change log

- **1.0 — 2026-11-09:** freezes accumulated requirements after a ranking-visibility
  regression; restores score/rank/Top % as a non-negotiable primary display and
  makes analysis, source scope and release review explicit.
