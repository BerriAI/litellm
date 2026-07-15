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
| Non-negative spend (all providers, uncached traffic) | `test_anthropic_claude3_transformation.py`, `llm_cost_calc/utils.py` | partial | yes (`test_no_provider_logs_negative_spend`) |
| Negative cost from bedrock cache-token split (#25846) | `test_llm_cost_calc_utils.py::test_inconsistent_cache_usage_never_prices_negative`, `test_anthropic_claude3_transformation.py::test_bedrock_sse_wrapper_bills_cache_rate_when_only_message_start_has_breakdown` (both fail-before-fix proven) | unsupported (live) | no - not reproducible on commercial bedrock; see note below. Covered offline instead |
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
| Concurrent increments (one key, parallel writers) | `tests/spend_tracking_tests/test_spend_accuracy_tests.py` (burst) | partial | yes (`test_burst_of_concurrent_calls_loses_no_spend`) |

## Spend read endpoints (verification surface)

| Endpoint | Existing | Status | Live e2e |
|----------|----------|--------|----------|
| `/spend/logs` (request_id / api_key) | `test_spend_management_endpoints.py` | covered | yes (primary read path; `test_spend_logs_endpoint_returns_spend` asserts 200 + spend, never 5xx) |
| `/spend/calculate` | `local_testing/test_spend_calculate_endpoint.py` | covered | yes (`test_spend_calculate_returns_nonzero_cost`) |
| `/spend/tags` | `test_spend_management_endpoints.py` | partial | yes (tag accuracy test) |
| `/spend/logs/v2` pagination (total/total_pages/out-of-range) | `test_spend_query_optimization.py` | covered | yes (`test_spend_logs_v2_pagination_caps_pages_and_keeps_total`; filter takes the hashed token, not the raw key) |
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
| `test_no_provider_logs_negative_spend` | no provider (gemini/anthropic/openai/bedrock) writes spend < 0 across streaming/non-streaming chat + embeddings; each provider still logs a positive row (non-vacuous). Uncached traffic only, so it does not reach the #25846 cache-token path; `fail_before_fix=unproven` |
| `test_failure_call_writes_failure_status_row` | failed call -> `status=failure`, `spend=0` |
| `test_spend_calculate_returns_nonzero_cost` | cost-map smoke (no batch wait) |
| `test_spend_logs_endpoint_returns_spend` | `/spend/logs` returns 200 + the key's spend, never a 5xx (intermittent-500 regression) |
| `test_burst_of_concurrent_calls_loses_no_spend` | N parallel calls on one key: N distinct costed rows, key aggregate == sum (no lost increments) |
| `test_spend_logs_v2_pagination_caps_pages_and_keeps_total` | `/spend/logs/v2` page cap, stable total on out-of-range page, zero total on no-match filter |
| `test_spend_routes.py` (23) | no spend route 404s or 5xxs |

## Why the #25846 cache-token path is `unsupported` live, not a gap

Probed against real bedrock (`us.anthropic.claude-haiku-4-5`, commercial `us-east-1`) on
2026-07-14 with a 14713-token cached prefix, streaming `/v1/messages`, cache write then
cache read. Findings, so nobody re-runs this:

- `message_stop` carries **no** cache breakdown here, only `{input_tokens: 12, output_tokens: N}`.
  So the "cache fields land on message_stop" reading of the code comment does not hold on
  commercial bedrock.
- The raw `message_delta` **already** carries the full breakdown
  (`input_tokens: 12, cache_read_input_tokens: 14713`), so it is self-consistent on its own.
  `prompt_tokens` becomes `12 + 14713 = 14725` and `text_tokens = 14725 - 14713 = 12`, never
  below zero.
- Confirmed by reverting the fix (`patched_stream = completion_stream`) and re-driving the
  same cached call: spend stayed **positive** (0.00166463, 0.00165363; prompt_tokens 14725),
  zero negative rows. `_merge_message_start_cache_into_delta_usage` is a no-op on this
  deployment.

The bug needs a deployment whose `message_delta` omits the cache breakdown (GovCloud, per the
transformation docstring / LIT-2411). A live commercial call cannot produce that payload, so
this cell is `unsupported` for live e2e (excluded from the denominator) rather than a gap to
chase. It is covered offline instead by the two tests named above, both proven fail-before-fix
by removing the real fix. Revisit only if GovCloud is in scope.

## Design + timing

`proxy_batch_write_at` (~60s) means rows land late; every read polls to a deadline.
Fresh scoped key per test (isolation, xdist-safe, cleaned up). Assert invariants
(`spend > 0`, `total == prompt + completion`, aggregate == sum), not literal
$/token values, so pricing drift is not a failure. Skip on environment (no proxy /
no provider key), fail on behavior (a real 2xx call with a wrong/missing row).
