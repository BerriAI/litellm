Unit tests for individual LLM providers.

Name of the test file is the name of the LLM provider - e.g. `test_openai.py` is for OpenAI.

## VCR-backed tests

Files matching `*_vcr.py` (e.g. `test_anthropic_completion_vcr.py`) replay
recorded HTTP traffic from `cassettes/` instead of calling the real provider.
They run offline by default — no API keys required, no per-PR cost.

To re-record against the live API:

```bash
ANTHROPIC_API_KEY=sk-ant-... \
  make test-llm-translation-record FILE=test_anthropic_completion_vcr.py
```

See [`cassettes/README.md`](./cassettes/README.md) for the full workflow,
including how to add a new cassette-backed test and what to scrub from
recordings before committing.
