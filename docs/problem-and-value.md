# From radar classification to an actionable review queue

Offshore operations need information about potential ice hazards when direct observation is difficult. The [original challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) asks for a probability that an already supplied target image is an iceberg. That is a useful component of surveillance, but it does not by itself locate objects across a complete scene or decide a vessel's route.

[C-CORE's published workflow](https://c-core.ca/solutions/maritime-surveillance/) combines automated analysis with human quality control and reporting. This provides a practical reference for the solution's potential role: provide consistent candidate scores so an analyst can prioritize review and pass verified information to operational planning. This is a proposed application of the model, not a measured customer deployment.

## The modeling decision that matters

The controlled study found that adding angle metadata reduces overall OOF loss from 0.22030 to 0.20479, but 95% of that improvement comes from 133 missing-angle ships. Test observations have known angles. Looking at the intended inference population therefore changes the interpretation of the gain.

On known-angle observations, the small metadata benefit reverses sign across training seeds. A better-supported addition is four-flip inference: it improves the known-angle ensemble for both image-only and metadata architectures, with the same saved weights. Its inference cost is four forward passes. [Measured comparison](modeling-study.md).

For an analyst-review application, model selection should follow the population the service will receive. The broader question is whether that predictive improvement changes the analyst's workload while retaining the required detection performance.

## From ML metric to an operating measure

| Level | Measure | Current evidence |
|---|---|---|
| Model probabilities | Log loss; class/metadata strata; seed/fold results | Measured in this repository |
| Review policy | Iceberg recall at a fixed false-alert or review budget | Proposed pilot measure; no threshold selected from operating costs |
| Workflow | Review time per candidate, turnaround, unresolved cases | Requires a representative analyst pilot |
| Operational value | Earlier usable information or reduced review effort while meeting safety policy | Hypothesis; no monetary or safety outcome claimed |

A minimal pilot would retain geographic/time acquisition identifiers, collect representative candidates and analyst outcomes, set a review policy with the operator, and run the model in shadow mode. Evaluate sensitivity by acquisition conditions and report missed hazards and false alerts separately. An exact incidence angle is only the repository's grouping proxy, not proof of independent satellite scenes.

The packaged inference contract must retain channel order, fitted scaling, optional-angle preprocessing and the chosen flip policy. A serving endpoint is useful here because the same tested prediction function can be called by a review application. Serving acceptance establishes that predictions can be consumed correctly; it does not establish a safety decision's adequacy. The Canadian Ice Service's [observation-to-bulletin workflow](https://www.canada.ca/en/environment-climate-change/services/ice-forecasts-observations/latest-conditions/products-guides/iceberg-bulletin-overview.html) illustrates the additional information chain around a classification result.
