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

## Layout

We use [`pytest-recording`](https://github.com/kiwicom/pytest-recording),
which auto-resolves the cassette path from the test location:

```
tests/llm_translation/
  cassettes/
    <test_module>/
      <test_name>.yaml
  test_<provider>_completion_vcr.py
  conftest.py        # provides the shared vcr_config fixture
```

For example, a test
`tests/llm_translation/test_anthropic_completion_vcr.py::test_basic` is backed by
`tests/llm_translation/cassettes/test_anthropic_completion_vcr/test_basic.yaml`.

## Adding a new cassette-backed test

1. Pick a small, deterministic call. Avoid prompts whose output depends on
   wall-clock time, randomness, or live web data.
2. Write the test as you normally would and decorate it with
   `@pytest.mark.vcr`. No imports beyond `pytest` are needed — the
   `vcr_config` fixture in `conftest.py` is applied automatically.
3. Run the sweep recorder once with the credentials you need. Recording is
   strictly opt-in via `--record-mode=once`; the default replay mode never
   touches the network.

## Bulk re-record (the common path)

A single sweep replays every `@pytest.mark.vcr` test under
`tests/llm_translation`, hitting the live provider only for tests that don't
yet have a cassette:

```bash
ANTHROPIC_API_KEY=sk-ant-... \
OPENAI_API_KEY=sk-... \
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
  make test-llm-translation-record
```

Or scope it to a single file:

```bash
ANTHROPIC_API_KEY=sk-ant-... \
  make test-llm-translation-record TARGET=test_anthropic_completion_vcr.py
```

vcrpy's `once` record mode does **not** overwrite an existing cassette —
delete the file first if you're intentionally refreshing it:

```bash
rm tests/llm_translation/cassettes/test_anthropic_completion_vcr/test_basic.yaml
ANTHROPIC_API_KEY=sk-ant-... \
  make test-llm-translation-record TARGET=test_anthropic_completion_vcr.py
```

To force a full refresh of every cassette in one shot:

```bash
rm -rf tests/llm_translation/cassettes/test_*
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... AWS_* \
  uv run pytest tests/llm_translation -m vcr --record-mode=all -v
```

## Refreshing the canned Anthropic fixtures (no API key)

The two Anthropic cassettes shipped with this directory are recorded against
an in-process mock so contributors can regenerate them without an
`ANTHROPIC_API_KEY`:

```bash
uv run python tests/llm_translation/cassettes/_record_anthropic_fixtures.py
```

For a full refresh against the real API, delete the cassettes first and use
the bulk-record sweep above.

## Cassette hygiene

After recording, **always inspect the YAML before committing**:

- The `vcr_config` fixture in `conftest.py` already filters the common
  request headers (`Authorization`, `x-api-key`, `anthropic-api-key`, AWS
  sigv4 headers, cookies, GCP keys, …) and per-request response headers
  (`set-cookie`, `cf-ray`, request IDs, org IDs, dates).
- A request *body* might still contain a token if your test passed one
  inline — scrub it manually.
- Quick sanity check: `grep -i 'sk-\|bearer\|api-key' cassettes/<dir>/*.yaml`
  should be clean.
- Trim unhelpful response bodies if they're megabytes large but the
  assertion only needs a few fields.

## Don't

- Don't commit cassettes with real API keys, OAuth tokens, or PII.
- Don't rely on cassettes for tests of *non-deterministic* behavior
  (rate-limit retries, timeouts, model creativity). Mock those at the
  LiteLLM layer instead.
- Don't manually edit cassette YAML beyond scrubbing — the format is
  byte-sensitive (e.g. content-length headers must match the body).
