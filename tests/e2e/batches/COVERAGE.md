# Batches Test Coverage Matrix

Live e2e coverage of the Batches API over a real proxy, real provider keys, and
real cost. Synchronous tier only: a batch's completion window is 24h, so these
tests never wait for `completed`. They assert the proxy accepts, routes, retrieves,
cancels, and lists a batch; everything created is deleted on teardown.

## Provider x operation

Only supported cells are tested. The capability table in `capabilities.py` holds one
row per supported (provider, scenario) pair, so there are no skipped cells in the
parametrized run. The batches suite never skips: missing provider creds or upstream
failures are hard test failures (see `tests/e2e/CLAUDE.md`).

| Provider  | create | retrieve | cancel | list | file backing |
|-----------|--------|----------|--------|------|--------------|
| OpenAI    | yes | yes | yes | yes | OpenAI Files |
| Azure     | yes | yes | yes | yes | Azure Files |
| Vertex AI | yes | yes | yes | yes | GCS (`gcs_bucket_name` / `GCS_BUCKET_NAME` on model) |
| Bedrock   | yes (`encoded` + `unified`) | yes | no (limited upstream) | no | S3 (`s3_bucket_name` + `aws_*` + `AWS_BATCH_ROLE_ARN` on model) |

Bedrock cancel is unreliable upstream and list is unsupported, so both are gated off
(`can_cancel=False`, `can_list=False`) when that provider is enabled in the matrix.
Bedrock file upload requires a model on the request (`encoded` / `unified` scenarios only);
`model_param` and `provider_fallback` are omitted because `POST /bedrock/v1/files` has no
model-less passthrough path, and Bedrock `create_batch` itself has no model-less path:
without a model it raises `LiteLLM doesn't support custom_llm_provider=bedrock for
'create_batch'`, since the model is what resolves the region, batch config, and IAM role
ARN. The `encoded` and `unified` scenarios both recover that model at create time (from the
model-scoped file id, or the managed file's `target_model_names`), so both route into the
Bedrock job-creation path; a raw file created model-less cannot.

## Routing scenarios (per `litellm/proxy/batches_endpoints/endpoints.py`)

Each create-capable provider runs all four. The test asserts the returned file id
and batch id carry the shape that scenario must produce (`matches_id_shape`):

| Scenario | How the batch is routed | File id | Batch id |
|----------|-------------------------|---------|----------|
| `encoded` | upload with `?model=` -> model-encoded file id -> create with just that id | model-encoded | model-encoded |
| `unified` | upload with `target_model_names=` -> unified managed file id -> create with that id | managed | managed |
| `model_param` | raw file (provider-fallback upload) -> create with `model` in the body | raw | model-encoded |
| `provider_fallback` | raw file -> `POST /{provider}/v1/batches`, env creds, no model | raw | raw (native provider shape) |

"managed" ids base64-decode to a `litellm_proxy` marker; "model-encoded" ids keep the
provider prefix and base64-encode `litellm:<id>;model,<model>`; "raw" ids are the
provider's native ids. Asserting these catches a proxy that returns a raw id where it
should manage it, or vice versa. On top of the id shape, a misroute to the wrong
provider also fails create (the file id / model do not belong there), and the
`provider_fallback` raw batch id is additionally checked against the provider's native
shape (`raw_id_matches_provider`).

## Key model restriction

`test_batch_key_model_access_denied` mints a key restricted to one model
(`resources.key(models=[...])`) and proves the proxy returns 403
`key_model_access_denied` both when that key uploads a file for a disallowed model
(files endpoint) and when it creates a batch for a disallowed model (batches
endpoint).

## Per-endpoint output assertions

Each endpoint's full response is validated, not just the id. File upload asserts
`object=="file"`, `purpose=="batch"`, a positive `bytes`, a status, and a created-at.
Batch create / retrieve assert `object=="batch"`, `endpoint=="/v1/chat/completions"`,
`completion_window=="24h"`, a non-empty `input_file_id`, and a created-at; retrieve
additionally cross-checks that `id` and `input_file_id` match the created batch.
Cancel asserts the same id, `object=="batch"`, and a cancelling/cancelled status. List
asserts the `object=="list"` envelope and that the created batch is present as a batch.
File delete asserts `object=="file"` and `deleted==True`.

## This suite's files

| File | Covers |
|------|--------|
| `batch_client.py` | typed file upload/download + batch create/retrieve/cancel/list/delete over the shared ProxyClient; runtime batch model registration via /model/new; denial helpers |
| `capabilities.py` | the provider x scenario matrix + per-provider /model/new params + id-shape classifiers + per-provider raw-id assertion |
| `conftest.py` | session-scoped batch deployment registration and teardown |
| `test_batches_e2e.py` | parametrized lifecycle with per-endpoint output assertions, file upload/delete outputs, key-model-access denial |

## Out of scope (intentionally)

Driving a batch to `completed`, cost tracking on completion, and the DB write-back
are not covered here; the 24h window makes them unfit for a synchronous gate. That
logic belongs in a DI-stubbed proxy integration test under `tests/test_litellm/proxy/`
where the provider client is injected to return `completed` deterministically.
