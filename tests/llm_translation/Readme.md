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

### Required environment

`REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` — same vars CircleCI uses for
its other Redis-backed jobs. Provider credentials
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AWS_*`, etc.) are needed only on
cache-miss (the daily re-record), not on replay.

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
