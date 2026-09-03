# Batches Test Coverage Matrix

Live e2e coverage of the Batches API over a real proxy, real provider keys, and
real cost. Mostly synchronous tier: a batch's completion window is 24h, so the
lifecycle matrix never waits for `completed`. It asserts the proxy accepts, routes,
retrieves, cancels, and lists a batch; everything created is deleted on teardown.
The exception is `TestBatchTerminalState`, which covers the completed state and
cost write-back via a cross-run marker baton (design below).

## Provider x operation

Only supported cells are tested. The capability table in `capabilities.py` holds one
row per supported (provider, scenario) pair, so there are no skipped cells in the
parametrized run. The batches suite never skips: missing provider creds or upstream
failures are hard test failures (see `tests/e2e/CLAUDE.md`).

| Provider  | create | retrieve | cancel | list | content download | file backing |
|-----------|--------|----------|--------|------|------------------|--------------|
| OpenAI    | yes | yes | yes | yes | yes (lifecycle + terminal output) | OpenAI Files |
| Azure     | yes | yes | yes | yes | yes (byte-verbatim) | Azure Files |
| Vertex AI | yes | yes | yes | yes | yes (provider-transformed) | GCS (`gcs_bucket_name` / `GCS_BUCKET_NAME` on model) |
| Bedrock   | yes (unified only) | yes | no (limited upstream) | no | yes (provider-transformed) | S3 (`s3_bucket_name` + `aws_*` + `AWS_BATCH_ROLE_ARN` on model) |

Bedrock cancel is unreliable upstream and list is unsupported, so both are gated off
(`can_cancel=False`, `can_list=False`) when that provider is enabled in the matrix;
flipping those gates is tracked in LIT-4774 and deliberately not part of this suite.
Bedrock file upload requires a model on the request (`encoded` / `unified` scenarios only);
`model_param` and `provider_fallback` are omitted because `POST /bedrock/v1/files` has no
model-less passthrough path.

`GET /v1/files/{id}/content` is exercised for the unified upload path per backend in
`test_unified_file_content_downloads`. Azure stores the JSONL verbatim, so its download
is asserted byte-equal to the upload. Vertex (GCS) and Bedrock (S3) transform lines at
upload time, so those assert a 200 with non-empty parseable JSON lines instead. Gemini
(non-Vertex) raises `NotImplementedError` for file content and has no cell here.

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
| `test_batches_e2e.py` | parametrized lifecycle with per-endpoint output assertions, file upload/delete outputs, key-model-access denial, per-backend content download, failure paths, second-hop routing, terminal state + cost |
| `test_managed_files_enforcement_e2e.py` | require_managed_files enforcement pins; deselected unless `E2E_MANAGED_FILES_STACK` is set (see below) |

## require_managed_files enforcement (separate stack phase)

`litellm_settings.require_managed_files` is a boot-time module global with no per-key
or runtime override, and turning it on 400s every upload that lacks
`target_model_names`, including the files_settings-routed `provider_fallback`
scenario above. So its pins cannot share a proxy with the rest of this suite:
`test_managed_files_enforcement_e2e.py` carries the `managed_files` marker, is
deselected unless `E2E_MANAGED_FILES_STACK` is set (the same pattern as the `weekly`
marker), and the PR gate runs it in a sequential phase after the main suite, against
the same ephemeral stack redeployed with the flag on. The pins: upload without
`target_model_names` is a 400, upload carrying a `model` param is a 400, a raw
provider file id on retrieve is a 400, and another user's managed unified id is a
403 while the owning user still retrieves it.

## Failure paths

`TestBatchFailurePaths` pins the customer-facing error contracts. A malformed input
file is a 400 at upload naming the bad content. A JSONL line whose url contradicts
the batch endpoint passes create (providers validate asynchronously) and drives the
batch to `failed` with structured `errors.data` (code/line/message), a null
`output_file_id`, and a $0 spend row keyed `{batch_id}_batch_cost` (LIT-4852: a
failed batch books $0 instead of crashing cost tracking). Cancelling that failed
batch is a 409 naming the terminal status. A file id encoded for one deployment wins
over a conflicting `model` param on create: the batch routes and re-encodes by the
file's embedded model (foreign-id precedence).

## Second hop (two chained gateways)

`TestBatchSecondHop` registers a `litellm_proxy/<inner model>` deployment pointing at
the proxy's own base URL with a freshly minted virtual key, so unified upload and
create traverse gateway -> gateway -> OpenAI (LIT-5347, PR #36240). The pin:
`target_model_names` is rewritten to the inner deployment on the second hop and the
nested managed ids round-trip retrieve. This self-chaining only needs the proxy to
reach its own `PROXY_BASE_URL`, which holds both locally and on the e2e stage.

## Terminal state + cost write-back (cross-run marker baton)

The 24h completion window rules out submit-and-wait inside one run, so
`TestBatchTerminalState` amortizes across runs. Each run submits a 1-line marker
batch (stable metadata key/value plus a per-run field) and deliberately never
cancels or deletes it or its input file: the marker is the baton the next run picks
up (OpenAI files expire on their own after ~30 days). Polling is list-only, up to 5
minutes, because retrieving a non-terminal batch books a $0 spend row whose
request_id then blocks the later real-cost row (`skip_duplicates`); the single
retrieve happens only once a completed marker exists. The assertion target is the
newest completed marker from ANY run: run-scoped deployment names mean the list
re-encodes prior-run batches under new encoded ids, so their spend keys are fresh
and a prior-run marker is billable by this run. On the 6h stage cadence the full
assertions are therefore deterministic from run 2 onward. On a cold start (no
completed marker within the poll budget) the test passes on the submission
assertions alone: a documented vacuous pass, not a skip. Markers aged past the 24h
window (25h-73h band, within the newest 100-item list page) must be terminal.

The cost assertion is the LIT-5730 headline: retrieving a completed model-encoded
batch must write a positive spend row with call_type `aretrieve_batch` and token
usage. Before the fix in `litellm/batches/batch_utils.py`, the retrieve endpoint
re-encoded the response's `output_file_id` in place before the queued logging
worker ran, the worker sent that encoded id to OpenAI, got a 404, and the spend row
never landed.

## Out of scope (intentionally)

Unified (managed) batch cost is owned by the hourly `CheckBatchCost` poller, and a
terminal DB status short-circuits retrieve for those ids, so the terminal-state cell
uses the encoded path; poller timing does not fit an e2e gate and belongs in a
DI-stubbed proxy integration test under `tests/test_litellm/proxy/`. Bedrock
cancel/list stay gated pending LIT-4774. Gemini (non-Vertex) file content raises
`NotImplementedError` upstream and is not a coverage cell.
