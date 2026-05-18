Unit tests for individual LLM providers.

Name of the test file is the name of the LLM provider - e.g. `test_openai.py` is for OpenAI.

## Layered VCR cache (L1 disk + L2 Redis or S3/R2)

Every test in this directory is auto-decorated with `@pytest.mark.vcr` (via
`conftest.py`). The first time a test runs we hit the live provider and
record the HTTP exchange into the cassette store under
`litellm:vcr:cassette:<test_id>`. Every subsequent run within 24h replays
from cache without touching the network. The 24h TTL means each new day's
first run records again, so upstream API drift surfaces within a day.

The store is layered:

1. **L1 — local disk.** When `CASSETTE_LOCAL_CACHE_DIR` is set (CircleCI
   wires this up automatically through the `restore_vcr_l1_cache` /
   `save_vcr_l1_cache` commands) the persister reads through a
   sharded directory cache before touching the remote backend. Most
   re-runs hit L1 on the same shard and never produce L2 traffic.
2. **L2 — Redis or S3/R2.** Selected by env var precedence:
   - `CASSETTE_S3_BUCKET` (with optional `CASSETTE_S3_ENDPOINT` for
     non-AWS endpoints like Cloudflare R2) → S3-compatible object store,
     read TTL enforced via `LastModified`. Recommended: zero egress on
     R2 means cassette reads are effectively free.
   - `CASSETTE_REDIS_URL` → Redis (legacy default).

Cassette payloads are zstd-compressed by default (~7-10x smaller than
the raw YAML). Set `CASSETTE_DISABLE_COMPRESSION=1` to opt out for
debugging. The loader sniffs the zstd frame magic, so cassettes
recorded before this rolled out keep working unchanged.

The persister, header scrubbing, and 2xx-only filtering are defined in
`tests/_vcr_redis_persister.py`. Files that already use `respx` (which
patches the same httpx transport vcrpy does) are excluded from the
auto-marker — see `_RESPX_CONFLICTING_FILES` in `conftest.py`.

The same VCR cache is used by other test directories that exercise live
provider APIs. The reusable conftest plumbing lives in
`tests/_vcr_conftest_common.py` and is wired into:

- `tests/llm_translation/`
- `tests/llm_responses_api_testing/`
- `tests/audio_tests/`
- `tests/batches_tests/`
- `tests/guardrails_tests/`
- `tests/image_gen_tests/`
- `tests/litellm_utils_tests/`
- `tests/local_testing/` (covers `local_testing_part1`, `local_testing_part2`,
  `litellm_router_testing`, `litellm_assistants_api_testing`,
  `langfuse_logging_unit_tests`)
- `tests/logging_callback_tests/`
- `tests/pass_through_unit_tests/`
- `tests/router_unit_tests/`
- `tests/unified_google_tests/`

Test directories that run LiteLLM proxy in Docker (e.g. `build_and_test`,
`proxy_logging_guardrails_model_info_tests`, `proxy_store_model_in_db_tests`)
are intentionally not included: VCR.py patches the in-process httpx
transport, so it cannot intercept the LLM calls that originate inside the
Docker container.

### Required environment

One of the L2 backends must be configured:

- `CASSETTE_S3_BUCKET` (preferred for new setups; pair with
  `CASSETTE_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com` and
  `CASSETTE_S3_REGION=auto` for Cloudflare R2 — zero egress fees), or
- `CASSETTE_REDIS_URL` — separate Redis instance from the application
  Redis (`REDIS_URL`/`REDIS_HOST`) so test cassettes are not flushed by
  proxy tests.

Optional:

- `CASSETTE_LOCAL_CACHE_DIR` — enables the L1 disk cache. CircleCI
  jobs set this automatically; for local dev it's normally fine to
  leave unset.
- `CASSETTE_DISABLE_COMPRESSION=1` — opt out of zstd compression for
  debugging.

Provider credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AWS_*`,
etc.) are needed only on cache-miss (the daily re-record), not on
replay.

### Flushing the cache

When you want the next run to re-record immediately instead of waiting
for the 24h TTL:

```bash
make test-llm-translation-flush-vcr-cache
```

### Disabling VCR

Skip the cache entirely (every call goes live, no recording):

```bash
LITELLM_VCR_DISABLE=1 uv run pytest tests/llm_translation/test_<file>.py
```
