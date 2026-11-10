# Requirements audit

## Presentation update — standard 1.1

The current primary display pairs **Public LB** and **Private LB** scores with
**Public Top %**. Rank/team counts and ties remain in detailed source evidence;
they are no longer repeated in the README table or GitHub description. Private
scores do not inherit public percentiles. This supersedes the canary-2 presentation
format reviewed below; its historical regression and source-preservation review
remain dated evidence. Local reproducibility may use scientific functions, a
notebook, script or existing thin command entry point; no bespoke CLI is required.
The existing canary-2 releases and completed merge/branch cleanup remain intact.
The original profile/standard are retained at `v0.2.0-canary.2`; the current
profile receipt records the latest approved compact description.


Audited **2026-11-09** against [portfolio standard 1.1](portfolio-standard.md).
The standard contains the accumulated requirements; this audit records their
project-specific applicability, evidence and remaining work. **Pass** means the
cited scope is supported. **Open** is accepted work, not a waiver. Older execution
receipts retain their original source identity and do not become fresh tests of
this editorial correction.

## Ranking regression and remediation

At `cdcca1e9de6e7d328bdff5b52662f48e7f413b13`, the README's historical/current
public-score and Top-% paragraph was on **line 5**. In the solution-canary
editorial state, `bea08768b13f9ec90dac257c11d055e402ab076b`, the same paragraph
was on **line 38**, below the opening and model discussion. The numbers survived,
but the required primary visibility did not. The domain-focused GitHub description
also dropped score/rank/Top-% context during the branding change.

The canary-2 correction restores a primary **Results** section immediately after
the opening, with historical scores **0.19348** and **0.20507** and modern score
**0.21463**, metric/direction, matched reference positions, denominator, Top %,
ties and snapshot. The live description was read back during this audit with
historical **1,657/3,330 (Top 49.76%)** and modern **1,980/3,330 (Top 59.46%)**,
explicitly labelled public score-equivalent ranks. These are not awarded final
ranks. OOF/holdout ranks are not applicable; a private rank is unavailable without
a matching private population. See [reference evidence](leaderboard-context.json)
and [detailed results](results.md).

## Acceptance matrix

| ID | Applicability | Status and project evidence |
|---|---|---|
| R01 | Full portfolio requirements apply. | **Pass for the review artifact:** this matrix covers R01–R26 and links the versioned [standard](portfolio-standard.md). Open execution/release items remain visible below. |
| R02 | A radar-classification solution is the presentation unit. | **Pass:** README opening → primary Results → model, controlled comparison, runnable CLI and saved-model use. [Problem/value](problem-and-value.md), [runbook](reproduction.md) and [serving](serving.md) connect useful output to implementation. |
| R03 | Verified compact presentation | Primary README and description retain public/private scores and public Top %. Rank/N and ties stay in linked detailed evidence. |
| R04 | Verified source scope | The metric/direction and submission are identified; private scores are paired with the same public submission. Detailed evidence retains reference counts, ties and snapshot. |
| R05 | Public, private, holdout and grouped OOF are different claims. | **Pass:** README explicitly distinguishes reference positions from awards and marks unavailable/non-applicable ranks. [Results](results.md) separates historical submissions, modern holdout, private score and grouped CV. |
| R06 | Modern model decisions need controlled evidence. | **Pass:** [study](modeling-study.md) retains image-only/metadata and no-flip/four-flip comparisons, both seeds, every fold, effect direction, runtime cost and next hypothesis. A metadata aggregate gain is not presented as a universal improvement. |
| R07 | Project-specific high-loss cases and input/model path apply. | **Pass for retained-OOF diagnostics:** [error analysis](error-analysis.md) traces known/missing-angle input strata through the recorded model/TTA probabilities to high-loss cohorts and seed overlap. The [receipt](error-analysis.json) independently reconciles the full OOF metric. Pixel-level mechanisms and true scene-held-out evaluation remain unmeasured questions. |
| R08 | Split/ID integrity and acquisition-aware validation apply. | **Pass within recorded scope:** [CV protocol](cv.md), [CV receipt](cv-validation.json) and analyzer check grouped assignments, all 1,604 records once per seed, preprocessing state, label/ID coverage, parent/child hashes and independent metric arithmetic. Exact angle is a proxy, not a verified scene identifier; two seeds do not supply confidence intervals. |
| R09 | Hard-case mining cannot change the reported test. | **Pass:** the fixed ceiling-top-10% BCE cohort is diagnostic only. [Error analysis](error-analysis.md) retains all rows, avoids filtered-score claims and documents no fitting, relabelling, threshold search or promotion. |
| R10 | A real checkpoint must retain its full inference recipe. | **Pass for the retained single-model package:** [serving](serving.md), adapter and [canary receipt](canary-validation.json) bind producer/checkpoint/source, HH/HV order, saved scaling/angle preprocessing, CPU environment and recorded flips. This is not the 20-model grouped-CV ensemble. |
| R11 | Real native→package→fresh-workspace HTTP parity applies. | **Pass for canary.1 scope:** [receipt](canary-validation.json) records four real-input probabilities, maximum absolute difference **0**, tolerance **1e-7**, invalid-request HTTP **400**, package integrity and cleanup. It does not establish production ingress or container-image validation. |
| R12 | Endpoint input must be explicit. | **Pass:** [serving contract](serving.md) accepts two 5,625-value raw radar bands plus optional nullable angle; the package owns fitted preprocessing and inference augmentation. It is not a full-scene detector or navigation service. |
| R13 | CLI, Docker/Lab/dev container, notebook and offline Kaggle paths apply. | **Preserved with earlier evidence:** [runbook](reproduction.md), [Kaggle guide](kaggle.md), executed notebooks and [hosted receipt](hosted-validation.json). The correction does not claim a fresh training, Docker-Lab or Kaggle execution. |
| R14 | The client depends on generic infrastructure in one direction. | **Pass:** [consumer contract](consumer-contract.md) and [shared-services guide](mlops.md) keep experiment/package/environment ownership here. No platform client catalog is needed. |
| R15 | Shared-service deployment UX belongs to the infrastructure owner. | **Delegated applicability:** this solution exports a model and supplies a request/runtime contract; it does not install shared services. The shared [requirements](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/requirements.md) cover `./infra`, state protection and lifecycle behavior. |
| R16 | Renames must preserve live and historical identities. | **Pass for correction scope:** current changes update presentation/URLs, not experiments, buckets or runtime names. The retained canary explicitly reports `shared_services_modified=false`. That is scoped evidence, not a new inventory of all running services. |
| R17 | Private inputs, weights and logs stay outside public assets. | **Pass for inspected Git/release scope:** no tracked `.pt`/`.joblib` weights or raw train/test/OOF/submission payloads were found. [Canary.1 assets](https://github.com/cttsai1985/kaggle-statoil-iceberg-solution/releases/tag/v0.2.0-canary.1) contain source/package, lock, checksums and aggregate documents; the receipt states private assets were not published. |
| R18 | Evidence must retain exact source/lock/model identity. | **Pass for retained canary; correction checks pending closure:** canary export source `e96d2dd4570ffca1cdabee3f554996e24fd27a18` is recorded clean with lock/package/runtime digests. New [error analysis](error-analysis.json) carries its own analyzer/input/OOF hashes. The old canary is not described as a fresh run of this correction. The current package/lock candidate is `0.2.0.dev2`; its final-source CI and release closure are separate from the retained `0.2.0.dev1` HTTP evidence. |
| R19 | This client participates in the coordinated canary release requirement. | **Published client evidence:** `v0.2.0-canary.1` is a GitHub prerelease at `bea08768b13f9ec90dac257c11d055e402ab076b`. **Open for this correction:** complete its merge/release review and coordinate portfolio release closure. A prerelease does not imply production traffic splitting. |
| R20 | Names, narrative, attribution and legacy source apply. | **Pass for current presentation:** canonical name is `kaggle-statoil-iceberg-solution`; opening describes the ML solution. MIT attribution and `legacy/` are retained. The scientific-preservation scope is enumerated below. |
| R21 | Domain value needs primary evidence and honest business scope. | **Pass:** [problem/value](problem-and-value.md) cites the official challenge, C-CORE's human-QC workflow and Canadian Ice Service context. Analyst-review benefits are proposed uses; savings, safety outcomes and customer ROI are not measured here. |
| R22 | Production batch and online serving remain the target. | **Open P1:** package/HTTP usability has evidence; full production research/batch and online-operation acceptance remains in the shared [requirements](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/requirements.md). The target is not redefined as demonstration-only. |
| R23 | Hosting, SLOs, access, capacity, alerts and recovery need real decisions. | **Open P1:** no new hosting/SLO or workload-capacity claim is made. TLS/auth ingress, alert receipt, off-host backups, restoration and production rollout/rollback require scoped implementation and evidence. |
| R24 | Verified recovery, quality gates and bounded retries apply at the batch boundary. | **Preserved; full operational acceptance open P1:** [consumer contract](consumer-contract.md), tracking/recovery tests and the [recorded batch drill](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/batch-operations.md) retain verified-output reuse and gated delivery. This does not mean every model run was scheduled through that runner. |
| R25 | Deeper ML hypotheses should come from retained evidence. | **Implemented diagnostic; hypotheses open P1:** the high-loss cohort, class/angle slices and seed overlap are measured in [error analysis](error-analysis.md). Image-quality inspection, retained-prediction comparisons by angle/class and genuinely scene/time-held-out validation remain proposed work, not completed improvements. |
| R26 | URL/export/release coherence and safe branch hygiene apply. | **In progress until correction merge:** canonical URLs and export metadata are being aligned; immutable old receipts retain their original identities. Merge/release checks and deletion of only verified merged branches remain release-owner actions; active/unmerged work must be preserved. |

## Scientific-preservation comparison

Two comparisons were performed using `git show <commit>:<path>` and byte equality:

1. **Pre-canary → merged canary/editorial:**
   `cdcca1e9de6e7d328bdff5b52662f48e7f413b13` →
   `bea08768b13f9ec90dac257c11d055e402ab076b`.
2. **Merged canary/editorial → canary-2 correction working tree.**

The following enumerated files are byte-identical in both comparisons:

| Scope | Checked paths |
|---|---|
| Controlled model study | `docs/modeling-study.md`, `docs/modeling-study.json` |
| Recorded evaluation/runtime evidence | `docs/validation.json`, `docs/cv-validation.json`, `docs/hosted-validation.json` |
| Notebook content and retained outputs | `notebooks/01-method-and-rerun.ipynb`, `notebooks/02-controlled-ablations.ipynb`, `notebooks/kaggle/rerun.ipynb` |
| Data/model/training/CV implementation | `src/iceberg/data.py`, `src/iceberg/model.py`, `src/iceberg/runtime.py`, `src/iceberg/cv.py` |
| Scientific recipes | `configs/baseline.json`, `configs/metadata.json`, `configs/image-only-flips.json` |

`docs/canary-validation.json`, `docs/serving.md` and `src/iceberg/serving.py` were
added by the canary work. They are byte-identical between the merged canary and
the canary-2 correction. For inspectable anchors, the current SHA-256 values are:

- Modeling-study JSON: `e12c2a14962a3d48ec64e68f2d106c21c52dd0a5883caa811a292f01581ca06b`.
- Canary receipt: `1ced67759c888253acd0f0d356d3dafeaccbab71b21a065c9bcb63cc3c47a251`.
- Training/runtime module: `a7dcc940aa37e6c5ae694f6e6b01706e6bca63a1c4721fb639ce1fe645f14715`.

`docs/results.md` is **not** included in the byte-identical claim. The pre-canary
to canary diff changed its infrastructure link/label; the canary-2 correction also
adds explicit score/rank context. README, package/version/URL metadata, exporter
metadata and new portfolio/error-analysis files are likewise outside the preserved
file set. `LICENSE` and every tracked `legacy/` file were also compared byte-for-byte across both transitions and matched. New diagnostics supplement the old evaluation; they do not overwrite
its metric population or producer receipts. This audit does not compare every
untracked private artifact or claim a fresh model-training reproduction.

The published canary asset inventory was read through GitHub during this audit; release archives were not newly downloaded or unpacked. No new training or HTTP canary was executed by this requirements review.

## Zoom study follow-up — 2026-11-10

The optional training zoom policies change `data.py` and `runtime.py` in a new, source-pinned scientific study. Earlier byte-preservation statements above refer to the enumerated canary-2 comparisons, not these later module bytes. [Protocol](zoom-study-protocol.md), [results](zoom-study.md) and [aggregate evidence](zoom-study.json) retain all folds/seeds and independent reload checks. Default-function equivalence is measured separately on the same current stack; public/private leaderboard values remain unchanged. This does not close the production operations requirements.
