# E2E LLM Translation — Long-Running Agent Loop

Drives an agent (typically Claude Code on a Max subscription) that
strengthens the e2e test suite for one provider's translation code
until the suite reaches a coverage + mutation-survival bar. The goal of
the *test bar* is to enable a later refactor of the provider's source
code with confidence.

## Why e2e, not unit

If the tests assert on internals (private functions, class attributes,
specific helper signatures), then any refactor of those internals also
rewrites the tests — and a test rewritten in the same change as the
code it covers is no longer a check on that change. E2E tests pin only
the public contract (`completion()`, `acompletion()`, streaming
iterators, request shape sent on the wire, response shape returned to
the caller), which is what we promise to users. Internals stay free to
refactor.

## What the agent must do

1. Pick **one** provider at a time (`anthropic`, `openai`, `bedrock`,
   ...). The provider name maps to `litellm/llms/<provider>/` (source)
   and `tests/llm_translation/test_<provider>*.py` (tests).
2. Make `make e2e-llm-coverage PROVIDER=<provider>` pass with
   `FAIL_UNDER=90` (configurable upward).
3. Then make `make e2e-llm-mutation PROVIDER=<provider>` finish with
   mutation-survival ≤ 10% (i.e. ≥ 90% killed). Equivalent mutants —
   log-string tweaks, unreachable error-message text, dead branches —
   are acceptable survivors; document them in
   `tests/llm_translation/E2E_AGENT_NOTES.md` rather than chasing them.
4. Never edit `litellm/` source. The `make e2e-llm-lock` target enforces
   this with `chmod u-w`; the agent should treat any failed write under
   `litellm/llms/<provider>/` as a signal to write a test instead.
   `chmod` does not stop a process running as root, so additionally run
   `make e2e-llm-hook-install` once — this installs a local git
   pre-commit hook that rejects commits touching `litellm/` while the
   `.e2e-llm-loop-active` sentinel file is present.

## Rules for tests

- Use the existing Redis-backed VCR plumbing
  (`tests/_vcr_conftest_common.py`). New tests inherit it automatically.
- For files that need `respx` instead, add them to
  `_RESPX_CONFLICTING_FILES` in `tests/llm_translation/conftest.py`.
- **Never** record cassettes to disk. The repo uses Redis cassettes and
  `.gitignore` blocks file cassettes — if a test creates a `.yaml`
  cassette, treat that as a bug.
- Assert on the **request** the SDK puts on the wire (URL, headers
  modulo scrubbed auth, JSON body) and on the **response shape** the
  SDK returns. Avoid assertions on private attributes of provider
  classes.

## Operating loop (suggested)

```bash
# 1. Lock source so the agent can only touch tests.
make e2e-llm-lock

# 2. Run coverage; iterate until pass.
make e2e-llm-coverage PROVIDER=anthropic FAIL_UNDER=90

# 3. Run mutation; iterate until survival ≤ 10%.
make e2e-llm-mutation PROVIDER=anthropic

# 4. Unlock and refactor the provider's source. Tests must stay green.
make e2e-llm-unlock
#    ... edit litellm/llms/anthropic/ ...
make e2e-llm-coverage PROVIDER=anthropic FAIL_UNDER=90
```

## Cost note

Test runs replay from Redis VCR by default — no provider API spend
after first record. Re-record cost (24h TTL or
`make test-llm-translation-flush-vcr-cache`) is real and per-call.
Agent reasoning/edit cost is covered by the Max plan; the agent is the
expensive thing here, not the LLM API.

## Stop conditions

The loop ends when one of:

- Coverage ≥ `FAIL_UNDER` **and** mutation-survival ≤ 10% on the
  selected provider.
- The agent reports it cannot make further progress without source
  changes (the locked tree forces this signal — it can't silently work
  around a hard-to-test code path).
- The user calls a stop.
