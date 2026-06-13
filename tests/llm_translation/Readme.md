Unit tests for individual LLM providers.

Name of the test file is the name of the LLM provider - e.g. `test_openai.py` is for OpenAI.

## Redis-backed VCR cache

Every test in this directory is auto-decorated with `@pytest.mark.vcr` (via
`conftest.py`). The first time a test runs we hit the live provider and
record the HTTP exchange into Redis under
`litellm:vcr:cassette:<test_id>`. Every subsequent run within 24h replays
from Redis without touching the network. The 24h TTL means each new day's
first run records again, so upstream API drift surfaces within a day.

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

`CASSETTE_REDIS_URL` — separate Redis instance from the application
Redis (`REDIS_URL`/`REDIS_HOST`) so test cassettes are not flushed by
proxy tests. Provider credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`AWS_*`, etc.) are needed only on cache-miss (the daily re-record), not
on replay.

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
