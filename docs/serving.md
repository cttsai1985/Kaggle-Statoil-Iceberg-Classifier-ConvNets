# Using the saved Statoil Iceberg model

The canary packages an evaluated model and checks that its prediction function remains usable through MLflow's HTTP interface. The solution owns preprocessing, model semantics and acceptance criteria; [ML Infrastructure](https://github.com/cttsai1985/ml-infrastructure) provides the generic serving verifier.

## Prediction contract

- **Input:** band_1 and band_2 arrays, each with 5,625 finite values, plus optional numeric inc_angle.
- **Output:** is_iceberg from the retained CNN and its recorded flip-averaging policy.
- **Model identity:** the producer manifest/checkpoint and copied artifact hashes are retained separately from the serving package's source and digest.

Use `dataframe_split` JSON to preserve named columns. Keep request examples and expected predictions private. The export contains model code and fitted artifacts; it does not need the original training notebook or data checkout to predict.

## Export a completed run

```bash
uv sync --locked --group serving
.venv/bin/python scripts/export_serving.py \
  --run-dir /absolute/path/to/completed-iceberg-run --threads 2 --max-batch-rows 16 \
  --output .local/my-model
```

Use a new output path. Export does not retrain. The package includes an MLflow signature, model dependencies, copied prediction code and model artifacts. The serving runtime uses its own locked model environment; the tracking server does not install this solution's dependencies.

## Verify through the shared service function

Prepare a representative private request, its independently computed native predictions as `{"predictions": [...]}`, and an invalid request missing a required field. Run from the infrastructure checkout:

```bash
./infra model verify \
  --model /absolute/path/to/exported-model \
  --python /absolute/path/to/solution/.venv/bin/python \
  --request /absolute/path/to/request.json \
  --expected /absolute/path/to/expected.json \
  --invalid-request /absolute/path/to/invalid-request.json \
  --output .local/canary-check
```

The verifier copies the package into a fresh private workspace, runs MLflow outside the producer checkout, checks health, compares predictions and requires the invalid request to fail. Its receipt binds the model digest, runtime, request/expected hashes, prediction differences and cleanup. Request rows, model weights and server logs remain private.

This canary is a tested prerelease, not a split of live customer traffic. Endpoint authentication, production resource sizing, monitoring and rollout policy are selected by the infrastructure deployment; model quality and the business decision remain this solution's responsibility. [Problem and value](problem-and-value.md) · [Evaluation results](results.md).

## Verified canary

[The original canary receipt](canary-validation.json) retains its canary-1 source and runtime. [The zoom follow-up compatibility receipt](zoom-serving-validation.json) separately verifies the retained reference model with the canary-3 code: four real-input rows, exact native/HTTP parity, HTTP 400 schema rejection, package identity and cleanup. It does not promote a study checkpoint or replace the original evidence. The checked request sample is an interface/parity test; full model evaluation remains in [results](results.md).

After verification, use the same package with [MLflow's foreground serving command](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/serving.md#use-the-verified-model) to expose `/invocations` to an application. The canary verifier is intentionally bounded; the application endpoint has an explicitly managed lifetime.
