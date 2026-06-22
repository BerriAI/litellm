# Batches Test Coverage Matrix

Live e2e coverage of the Batches API over a real proxy, real provider keys, and
real cost. Synchronous tier only: a batch's completion window is 24h, so these
tests never wait for `completed`. They assert the proxy accepts, routes, retrieves,
cancels, and lists a batch; everything created is deleted on teardown.

## Provider x operation

Only supported cells are tested. The capability table in `capabilities.py` holds one
row per supported (provider, scenario) pair, so there are no skipped cells in the
parametrized run.

| Provider  | create | retrieve | cancel | list | file backing |
|-----------|--------|----------|--------|------|--------------|
| OpenAI    | yes | yes | yes | yes | OpenAI Files |
| Azure     | yes | yes | yes | yes | Azure Files |
| Vertex AI | yes | yes | yes | yes | GCS bucket |
| Bedrock   | yes | yes | no (limited upstream) | no | S3 bucket |
| Anthropic | no  | yes (env-gated) | no | no | Anthropic Files |

Bedrock cancel is unreliable upstream and list is unsupported, so both are gated off
(`can_cancel=False`, `can_list=False`). Anthropic cannot create/cancel/list through
litellm, so it has a standalone retrieve test that skips unless `ANTHROPIC_BATCH_ID`
points at a real Anthropic batch.

## Routing scenarios (per `litellm/proxy/batches_endpoints/endpoints.py`)

Each create-capable provider runs all four:

| Scenario | How the batch is routed | Routing assertion |
|----------|-------------------------|-------------------|
| `encoded` | upload with `?model=` -> model-encoded file id -> create with just that id | create succeeds against the provider's model (id is re-encoded, not provider-shaped) |
| `unified` | upload with `target_model_names=` -> unified managed file id -> create with that id | create succeeds against the provider's model |
| `model_param` | raw file (provider-fallback upload) -> create with `model` in the body | create succeeds against the provider's model |
| `provider_fallback` | raw file -> `POST /{provider}/v1/batches`, env creds, no model | raw batch id matches the provider's native shape (`raw_id_matches_provider`) |

For the three encoded scenarios the proxy re-encodes the returned batch id, so the id
itself is not provider-discriminating; the load-bearing signal is that create
succeeds against that provider's own model (a misroute to the wrong provider fails
because the file id / model do not belong there). Only `provider_fallback` returns a
raw id whose shape we can assert directly.

## Key model restriction

`test_batch_key_model_access_denied` mints a key restricted to one model
(`resources.key(models=[...])`) and proves the proxy returns 403
`key_model_access_denied` both when that key uploads a file for a disallowed model
(files endpoint) and when it creates a batch for a disallowed model (batches
endpoint).

## This suite's files

| File | Covers |
|------|--------|
| `batch_client.py` | typed file upload/download + batch create/retrieve/cancel/list over the shared Gateway; denial helpers |
| `capabilities.py` | the provider x scenario matrix + per-provider raw-id assertion |
| `test_batches_e2e.py` | parametrized lifecycle, key-model-access denial, anthropic retrieve |

## Out of scope (intentionally)

Driving a batch to `completed`, cost tracking on completion, and the DB write-back
are not covered here; the 24h window makes them unfit for a synchronous gate. That
logic belongs in a DI-stubbed proxy integration test under `tests/test_litellm/proxy/`
where the provider client is injected to return `completed` deterministically.
