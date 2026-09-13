Unit tests for individual LLM providers.

Name of the test file is the name of the LLM provider - e.g. `test_openai.py` is for OpenAI.

## VCR cassette cache

Every test in this directory is auto-decorated with `@pytest.mark.vcr` (via
`conftest.py`). The first time a test runs we hit the live provider and
record the HTTP exchange into the configured cassette backend. Every subsequent
run within the cassette lifetime replays without touching the network. The
default 24h lifetime means each new day's first run records again, so upstream
API drift surfaces within a day

The shared persister and filesystem backend are defined in
`tests/_vcr_persister.py`. The Redis backend and compatibility exports remain
in `tests/_vcr_redis_persister.py`. Files that already use `respx` (which
patches the same httpx transport vcrpy does) are excluded from the auto-marker,
see `_RESPX_CONFLICTING_FILES` in `conftest.py`

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

### Backend and lifetime

| Variable | Meaning | Default |
| --- | --- | --- |
| `CASSETTE_BACKEND` | Explicitly selects `redis` or `filesystem` | Redis when `CASSETTE_REDIS_URL` is set, otherwise filesystem |
| `CASSETTE_REDIS_URL` | Dedicated Redis URL, separate from application Redis | Unset |
| `CASSETTE_TTL_SECONDS` | Write-time lifetime in seconds; `0`, negative values, and `inf` never expire | `86400` |
| `LITELLM_VCR_DISABLE` | Set to `1` to disable VCR | Unset |

`CASSETTE_TTL_SECONDS` sets the lifetime for newly recorded cassettes and
defaults to `86400`. Values at or below zero, and `inf`, never expire. A test can
override the environment at write time:

```python
@pytest.mark.cassette_ttl(3600)
def test_short_lived_recording():
    ...
```

Redis stores the lifetime using `SET EX` and keeps its existing key format,
`litellm:vcr:cassette:<test_id>`. Filesystem cassettes use vcrpy's cassette path
and include `recorded_at` and `ttl_seconds` alongside `version` and
`interactions`. Provider credentials are needed only on a cache miss

A filesystem cassette is self-describing:

```yaml
version: 1
recorded_at: '2026-09-02T12:00:00+00:00'
ttl_seconds: 86400
interactions:
- request: {...}
  response: {...}
```

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
