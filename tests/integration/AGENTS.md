# tests/integration

Real proxy, Postgres, Redis, scripted upstream. `README.md` has shards and CI wiring

## What good looks like

The root example is from here. The spend row lands async: poll, never sleep

```python
rows = eventually(
    lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (body["id"],)),
    lambda values: len(values) == 1,
    seconds=70,
)
assert float(rows[0]["spend"]) == pytest.approx(10 * 0.001 + 5 * 0.0001 + 7 * 0.002 + 4 * 0.002)
```

`sleep(3)` fails on a slow runner and taxes every fast one. Assert the outbound body in the upstream
handler; a leaked field is invisible from the response. `monkeypatch.setenv` is fine; patching our own
function in a full stack is not

## Where it goes

By the domain a user would name: `pricing`, `spend`, `routing`, `mcp`. A file only needs to live in a
directory that a `GROUPS` entry in `run.py` selects; there is no manifest and no `covers` marker on new
tests. A product bug the test exposes is `pytest.skip("BUG: <symptom>")` at the top of the body, not a
fix in the test and not a deletion. Needs no proxy, DB or Redis: `tests/unit`

## Translation cases

Each model gets one complete base `TranslationTestCase` in `translation/<endpoint>/bases/<provider>.py`,
named `<MODEL>_TEST_CASE` after its deployment (`anthropic/claude-sonnet-4-6` is
`CLAUDE_SONNET_4_6_TEST_CASE`). A feature case is `<MODEL>_<SCENARIO>_TEST_CASE`. Import a base under its
own name, never aliased to `BASE`, so every case shows which model it derives from

```python
from integration.translation.messages.bases.anthropic import CLAUDE_SONNET_4_6_TEST_CASE

CLAUDE_SONNET_4_6_THINKING_BUDGET_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="thinking_budget",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "max_tokens": 2048, "thinking": ...},
    ...
)
```
