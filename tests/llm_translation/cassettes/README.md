# VCR cassettes for LLM translation tests

This directory holds [vcrpy](https://vcrpy.readthedocs.io/) cassettes used by
`tests/llm_translation/` to replay real provider HTTP traffic without hitting
the live API.

Why this exists is tracked in
[LIT-2683](https://linear.app/litellm-ai/issue/LIT-2683) and discussed in
`#sdlc` on Slack: e2e tests were repeatedly draining provider billing accounts
and producing flaky CI on outages. Recording the HTTP exchange once and
replaying it on subsequent runs gives us realistic provider responses
(streaming, headers, edge-case payloads) at zero per-PR cost.

## How to add a new cassette-backed test

1. Pick a small, deterministic call. Avoid prompts whose output depends on
   wall-clock time, randomness, or live web data.
2. Add a test in a `*_vcr.py` file under `tests/llm_translation/`. Wrap it
   with `@litellm_vcr.use_cassette("<some_name>.yaml")` from
   `tests/llm_translation/vcr_config.py`.
3. Record the cassette once:

   ```bash
   LITELLM_VCR_RECORD_MODE=once \
     ANTHROPIC_API_KEY=sk-ant-... \
     uv run pytest tests/llm_translation/test_my_provider_vcr.py::test_my_case -v
   ```

   or, equivalently:

   ```bash
   ANTHROPIC_API_KEY=sk-ant-... \
     make test-llm-translation-record FILE=test_my_provider_vcr.py
   ```

4. Inspect the resulting YAML file:
   - **Strip any secrets** that survived `vcr_config.py`'s header filter.
     `vcr_config.py` already removes the common ones (`Authorization`,
     `x-api-key`, `cookie`, AWS sigv4 headers, etc.) — but a request *body*
     might contain a token if your test passed one inline.
   - Trim very large response bodies if they aren't load-bearing for the
     assertion.
5. Commit the cassette alongside the test.

## Re-recording

Run the same `make test-llm-translation-record` command. vcrpy's `once` mode
will *not* overwrite an existing cassette — delete the file first if you're
intentionally refreshing it:

```bash
rm tests/llm_translation/cassettes/anthropic_basic_completion.yaml
ANTHROPIC_API_KEY=sk-ant-... make test-llm-translation-record \
    FILE=test_anthropic_completion_vcr.py
```

## Refreshing the canned Anthropic fixtures

The two Anthropic cassettes in this directory
(`anthropic_basic_completion.yaml` and `anthropic_streaming_completion.yaml`)
are recorded against an in-process mock so contributors can regenerate them
without an `ANTHROPIC_API_KEY`:

```bash
uv run python tests/llm_translation/cassettes/_record_anthropic_fixtures.py
```

For a full refresh against the real API, delete the cassettes first and use
the `LITELLM_VCR_RECORD_MODE=once` path with a real key.

## Don't

- Don't commit cassettes containing real API keys, OAuth tokens, or PII.
  When in doubt, `grep -i 'sk-\|bearer\|api-key' cassettes/*.yaml` after
  recording.
- Don't rely on cassettes for tests of *non-deterministic* behavior
  (rate-limit retries, timeouts, the model itself making a creative choice).
  Mock those at the LiteLLM layer instead.
- Don't record both real and mock host names into the same cassette without
  rewriting the URL — vcrpy matches on host/port by default.
