Unit tests for individual LLM providers.

Name of the test file is the name of the LLM provider - e.g. `test_openai.py` is for OpenAI.

## VCR-backed tests

Tests decorated with `@pytest.mark.vcr` (typically in `*_vcr.py` files,
e.g. `test_anthropic_completion_vcr.py`) replay recorded HTTP traffic from
`cassettes/` via [`pytest-recording`](https://github.com/kiwicom/pytest-recording)
instead of calling the real provider. They run offline by default — no API
keys required, no per-PR cost.

To re-record every marked test in one sweep:

```bash
ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... \
  make test-llm-translation-record
```

To scope to a single file:

```bash
ANTHROPIC_API_KEY=sk-ant-... \
  make test-llm-translation-record TARGET=test_anthropic_completion_vcr.py
```

See [`cassettes/README.md`](./cassettes/README.md) for the full workflow,
including how to add a new cassette-backed test and what to scrub from
recordings before committing.
