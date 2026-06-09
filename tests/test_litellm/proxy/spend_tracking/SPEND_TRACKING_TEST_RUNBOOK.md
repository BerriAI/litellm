# Spend Tracking Test Runbook

This runbook documents the regression coverage that guards litellm's spend/cost
tracking against silent breakage before a release. It has two halves: a set of
deterministic offline unit tests (Part A) that run in CI, and an operator-driven
live end-to-end check (Part B) that hits real provider APIs and asserts real
SpendLogs rows. Spend tracking is already heavily tested across the codebase
(cost calculators, SpendLogsPayload construction, the proxy cost callback, the DB
spend-update writer, daily-spend queues); the tests below fill the specific
high-value gaps that previously had no direct coverage.

## Part A: offline regression tests (CI)

Run them with:

```bash
uv run --no-sync python -m pytest \
  tests/test_litellm/proxy/spend_tracking/test_spend_tracking_utils.py \
  tests/test_litellm/proxy/hooks/test_proxy_track_cost_callback.py \
  tests/test_litellm/test_cost_calculator.py::test_embedding_completion_cost_uses_input_cost_per_token \
  -q
```

Each test is written to fail if the specific guard it covers is mutated; the
"What it guards" column names the source line that, when broken, turns the test
red. Verified by hand-mutation (revert each guard and watch the row go red).

| Test name | What it guards | Status |
| --- | --- | --- |
| `TestGetStatusForSpendLog::test_missing_status_key_defaults_to_success` | `_get_status_for_spend_log` default branch (spend_tracking_utils.py) returns success when no status set | pass |
| `TestGetStatusForSpendLog::test_explicit_success_returns_success` | same helper, explicit success preserved | pass |
| `TestGetStatusForSpendLog::test_failure_returns_failure` | the `== "failure"` guard so failed requests are logged as failures | pass |
| `TestGetStatusForSpendLog::test_non_failure_value_returns_success` | kills the "any non-None status -> failure" mutant | pass |
| `test_get_logging_payload_cache_hit_appends_unique_suffix_to_request_id` | cache-hit `_cache_hit{time}` suffix on request_id; without it SpendLogs hits duplicate-key collisions | pass |
| `test_get_logging_payload_failure_status_and_zero_spend` | status wiring at the `get_logging_payload` call site plus `spend` sourced from `response_cost` | pass |
| `test_get_logging_payload_default_status_success` | default status path through `get_logging_payload` | pass |
| `test_track_cost_callback_zeroes_response_cost_on_cache_hit` | cache-hit cost zeroing in `_PROXY_track_cost_callback`; the anti double-charge guard | pass |
| `test_embedding_completion_cost_uses_input_cost_per_token` | embedding cost = `prompt_tokens * input_cost_per_token`; previously no dedicated embedding cost test | pass |

All 9 pass on `claude/spend-tracking-tests-4gbezl`.

## Part B: live end-to-end check (operator-run, real spend logs)

This proves spend tracking against real provider responses and a real database,
which is the closest mirror of what a customer sees. It needs a Postgres
`DATABASE_URL`, an `OPENAI_API_KEY` in `.env`, and outbound access to
api.openai.com. It uses `gpt-5.4-nano` (current cheap small model as of 2026-06)
and `text-embedding-3-small`, with local caching enabled so a repeated identical
chat request produces a cache hit.

Config: `tests/test_litellm/proxy/spend_tracking/e2e_spend_config.yaml`

1. Point at a database and start the proxy (spend logs require a DB):

```bash
export DATABASE_URL='postgresql://user:pass@localhost:5432/litellm'
python litellm/proxy/proxy_cli.py \
  --config tests/test_litellm/proxy/spend_tracking/e2e_spend_config.yaml \
  --detailed_debug 2>&1 | tee litellm.log
```

2. First chat call (real cost expected):

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H 'Authorization: Bearer sk-1234' -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.4-nano","messages":[{"role":"user","content":"say hello in one word"}]}' | jq '{id, usage}'
```

3. Identical chat call again to trigger a cache hit (cost should be recorded as 0):

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H 'Authorization: Bearer sk-1234' -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.4-nano","messages":[{"role":"user","content":"say hello in one word"}]}' | jq '{id, usage}'
```

4. Embedding call (real cost expected):

```bash
curl -s http://localhost:4000/v1/embeddings \
  -H 'Authorization: Bearer sk-1234' -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-3-small","input":"hello world"}' | jq '{model, usage}'
```

5. Wait for the spend-log flush (the proxy batches writes roughly once a minute),
   then read the logs back and assert the three rows:

```bash
sleep 65
curl -s 'http://localhost:4000/spend/logs' -H 'Authorization: Bearer sk-1234' \
  | jq '[.[] | {request_id, call_type, model, spend, cache_hit}]'
```

Expected: the chat row has `spend > 0`, the embedding row has `spend > 0`, and the
cache-hit row has `spend == 0` with a `request_id` containing `_cache_hit`.

Status: not run in this sandbox. The container has no `.env`/`OPENAI_API_KEY`, no
`DATABASE_URL`, and egress to api.openai.com is blocked by the environment's
network policy (a billed call is not possible here). Run the steps above on a host
with credentials and network access to fill this status; the offline suite in
Part A is what gates CI
