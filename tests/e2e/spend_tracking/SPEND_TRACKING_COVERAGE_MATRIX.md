# Spend Tracking Test Coverage Matrix

Scope: every distinct spend-tracking code path, mapped to the test that exercises
it and the level it runs at. Highlights where a live e2e check is the only thing
that would catch a regression.

Companion: live suite `test_spend_tracking_e2e.py` + route breadth
`test_spend_routes.py` (this directory). Offline regression suite:
`tests/test_litellm/proxy/spend_tracking/`. Reference PR: BerriAI/litellm#29956.

Levels: `unit` mocked; `integration` real DB/cost-map; `live` real provider +
proxy + SpendLogs rows. Status: `covered` / `partial` / `gap`.

---

## SpendLogs row construction (`spend_tracking_utils.get_logging_payload`)

| Path | Existing | Level | Status | Live e2e |
|------|----------|-------|--------|----------|
| `_get_status_for_spend_log` | `test_spend_tracking_utils.py` | unit | covered | yes (status read off the row) |
| cache-hit `request_id` suffix | `test_spend_tracking_utils.py` | unit | covered | yes (`test_cache_hit_is_zero_cost_and_suffixed`) |
| failure status + zero spend | `test_spend_tracking_utils.py` | unit | partial | yes (`test_failure_call_writes_failure_status_row`) |
| per-model / per-provider attribution | `test_spend_tracking_utils.py` | unit | covered | yes (`test_each_model_on_a_shared_key_gets_its_own_row`) |
| field population (model/tokens/api_key/team/org) | `test_spend_tracking_utils.py` | unit | partial | yes (asserts real values) |
| `request_tags` propagation | `test_db_spend_update_writer.py` | unit | partial | yes (`test_request_tags_round_trip`) |
| `end_user` attribution | unit | unit | partial | yes (`test_end_user_spend_attributed_on_row`) |

## Cost calculation by modality

| Modality | Existing | Status | Live e2e |
|----------|----------|--------|----------|
| Chat (non-stream) | `test_cost_calculator.py`, `local_testing/test_completion_cost.py` | covered | yes (`test_chat_completion_writes_nonzero_spend_row`) |
| Chat (streaming) | `test_streaming_interrupt_spend_tracking.py` | partial | yes (`test_streaming_chat_completion_tracks_spend`) |
| Embedding | `test_cost_calculator.py` (#29956) | partial | yes (`test_embedding_writes_nonzero_spend_row`) |
| Pass-through (gemini/anthropic) | `pass_through_tests/*.test.js` + `llm_translation/` suite | covered | yes (llm_translation suite) |
| Image / audio / rerank / responses / realtime | per-provider unit cost tests | partial/gap | gap |

## Entity spend aggregation

| Entity | Existing | Status | Live e2e |
|--------|----------|--------|----------|
| API key | `test_db_spend_update_writer.py`, `test_spend_counters.py` | covered | yes (`test_key_spend_equals_sum_of_logs`) |
| Tag | `test_update_daily_tag_spend.py` | partial | yes (`test_tag_spend_matches_sum_of_tagged_logs`) |
| End-user | `test_proxy_update_spend.py` | covered | yes |
| Spend == sum(logs) consistency | none | gap | yes (key + tag aggregate == sum of rows) |

## Spend read endpoints (verification surface)

| Endpoint | Existing | Status | Live e2e |
|----------|----------|--------|----------|
| `/spend/logs` (request_id / api_key) | `test_spend_management_endpoints.py` | covered | yes (primary read path; `test_spend_logs_endpoint_returns_spend` asserts 200 + spend, never 5xx) |
| `/spend/calculate` | `local_testing/test_spend_calculate_endpoint.py` | covered | yes (`test_spend_calculate_returns_nonzero_cost`) |
| `/spend/tags` | `test_spend_management_endpoints.py` | partial | yes (tag accuracy test) |
| whole spend GET surface (22 routes) | unit per-handler | partial | yes (`test_spend_routes.py` probes each for 404/5xx) |

## What this suite pins

| Test | Invariant |
|------|-----------|
| `test_chat_completion_writes_nonzero_spend_row` | nonzero cost, token arithmetic, status, row findable by `response.id` |
| `test_streaming_chat_completion_tracks_spend` | streamed responses still costed |
| `test_embedding_writes_nonzero_spend_row` | embedding cost, `completion_tokens == 0` |
| `test_cache_hit_is_zero_cost_and_suffixed` | cache hits not double-charged; `_cache_hit` suffix |
| `test_key_spend_equals_sum_of_logs` | key aggregate == sum of rows |
| `test_request_tags_round_trip` | tags persist onto the row |
| `test_tag_spend_matches_sum_of_tagged_logs` | `/spend/tags` SUM/COUNT == tagged rows |
| `test_end_user_spend_attributed_on_row` | `end_user` attributed + costed |
| `test_each_model_on_a_shared_key_gets_its_own_row` | per-model/provider rows, correct model + cost, distinct request_ids matching response id |
| `test_failure_call_writes_failure_status_row` | failed call -> `status=failure`, `spend=0` |
| `test_spend_calculate_returns_nonzero_cost` | cost-map smoke (no batch wait) |
| `test_spend_logs_endpoint_returns_spend` | `/spend/logs` returns 200 + the key's spend, never a 5xx (intermittent-500 regression) |
| `test_spend_routes.py` (23) | no spend route 404s or 5xxs |

## Design + timing

`proxy_batch_write_at` (~60s) means rows land late; every read polls to a deadline.
Fresh scoped key per test (isolation, xdist-safe, cleaned up). Assert invariants
(`spend > 0`, `total == prompt + completion`, aggregate == sum), not literal
$/token values, so pricing drift is not a failure. Skip on environment (no proxy /
no provider key), fail on behavior (a real 2xx call with a wrong/missing row).
