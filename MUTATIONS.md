# Mutation evidence for litellm_sail_tests_integration

Review evidence for the tests/integration cell of the PR 42840 regression sweep. Not for merge

Tip: d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 (litellm_add_sail_provider). Base: 0c1c3e18d5250ec3a0e1e3f287e0b93e2906d900

Every run below used the existing integration harness against a real proxy (`LITELLM_LOCAL_MODEL_COST_MAP=True`,
`STORE_MODEL_IN_DB=True`, `num_retries: 0`), Postgres, Redis and the scripted upstream from
`tests/integration/_support/upstream.py`. Two stacks ran side by side: tip (upstream 8190, proxy 4000, database
`integration_tip`) and base (upstream 8191, proxy 4001, database `integration_base`). The proxy was restarted after
every apply and after every restore so the mutated module was actually the one serving

Runner used for every pytest invocation (paths shortened below as `run <tree> <label> <uport> <pport> <args>`):

```bash
export PATH=/home/ubuntu/repos/litellm/.venv/bin:$PATH
export PYTHONPATH="$tree:$tree/tests:$tree/tests/e2e"
export INTEGRATION_PROXY_URL="http://127.0.0.1:$pport"
export INTEGRATION_UPSTREAM_URL="http://127.0.0.1:$uport"
export INTEGRATION_MASTER_KEY=sk-integration-master
export DATABASE_URL="postgresql://postgres:postgres@127.0.0.1:5432/integration_$label"
export LITELLM_LOCAL_MODEL_COST_MAP=True
python -m pytest -p no:cacheprovider -q "$@"
```

## Tests

| File | Test | Covers |
| --- | --- | --- |
| tests/integration/pricing/test_service_tier_pricing.py | test_balanced_service_tier_bills_balanced_rates_and_keeps_pricing_off_the_wire | (2) `_balanced` rates via litellm_params bill balanced at `service_tier=balanced`, base with no tier, full upstream body |
| tests/integration/pricing/test_service_tier_pricing.py | test_balanced_and_flex_on_a_base_rate_only_model_bill_base_rates_and_pass_the_tier_through | (1) and (3) balanced and flex on a base-rate-only model bill base, upstream body carries the tier untouched |
| tests/integration/pricing/test_service_tier_pricing.py | test_extra_body_metadata_on_an_openai_shaped_deployment_replaces_the_wire_metadata_whole[openai] | (4) shallow overwrite of request metadata on a non-Sail openai deployment, full upstream body |
| tests/integration/pricing/test_service_tier_pricing.py | test_extra_body_metadata_on_an_openai_shaped_deployment_replaces_the_wire_metadata_whole[cognition] | (4) same through the JSON provider loader without special handling |
| tests/integration/pricing/test_service_tier_pricing.py | test_sail_flex_sends_completion_window_without_service_tier_and_bills_flex_rates | (5) `sail/<model>` sends `metadata.completion_window=flex` and no `service_tier`, bills `_flex`; explicit `asap` bills base; no tier bills base |
| tests/integration/pricing/test_responses_extra_body_pricing.py | test_responses_extra_body_on_an_openai_shaped_deployment_reaches_the_wire_and_bills_base_rates | (6) `/v1/responses` extra_body reaches the wire whole and bills base on a non-Sail deployment even when it carries `completion_window` |
| tests/integration/pricing/test_responses_extra_body_pricing.py | test_responses_extra_body_completion_window_on_a_sail_deployment_bills_the_window | (6) extra_body lands in Responses logging optional_params (observable as `_flex` billing on Sail, base on `asap`) |
| tests/integration/providers/test_sail_audio_transcription.py | test_audio_transcription_on_a_sail_deployment_is_rejected_before_the_upstream_sees_it | (7) skipped with `BUG:`, see Blockers |

Every helper asserts the whole upstream body with `==`, the `x-litellm-response-cost` header, then polls the
`LiteLLM_SpendLogs` row with `eventually(..., seconds=70)` and asserts `prompt_tokens`, `completion_tokens`, `spend`
and `cost_breakdown.input_cost` / `output_cost` to `rel=1e-6`. Rates are the test's own `litellm_params`, nothing is
read from the vendor cost map

## Tip run (all production files restored)

```
run /home/ubuntu/repos/litellm-sail-tip tip 8190 4000 tests/integration/pricing/test_service_tier_pricing.py tests/integration/pricing/test_responses_extra_body_pricing.py tests/integration/providers/test_sail_audio_transcription.py -v
tests/integration/pricing/test_service_tier_pricing.py ......            [ 66%]
tests/integration/pricing/test_responses_extra_body_pricing.py ..        [ 88%]
tests/integration/providers/test_sail_audio_transcription.py s           [100%]
======================== 8 passed, 1 skipped in 58.18s =========================
```

Final run on the committed files (after `ruff format` and the upstream observation scoping described under Gates):

```
run /home/ubuntu/repos/litellm-sail-tip tip 8190 4000 tests/integration/pricing/test_service_tier_pricing.py tests/integration/pricing/test_responses_extra_body_pricing.py tests/integration/providers/test_sail_audio_transcription.py
........s                                                                [100%]
8 passed, 1 skipped in 45.05s
```

## Base run (same test files copied into the 0c1c3e18 worktree, base proxy)

```
run /home/ubuntu/repos/litellm-sail-base base 8191 4001 tests/integration/pricing/test_service_tier_pricing.py tests/integration/pricing/test_responses_extra_body_pricing.py tests/integration/providers/test_sail_audio_transcription.py --tb=line
.F...F.Fs                                                                [100%]
E   AssertionError: ... assert 0.06 == 0.22 ± 2.2e-07   (test_service_tier_pricing.py:53, balanced tier billed base at base)
E   AssertionError: POST /model/new: 500 {"error":{"message":"Model create was saved to the database, but the model id(s) [...] are not live in this pod's router after the reload and are not being served by this pod. ..."}}   (sail/ deployment cannot be registered at base)
E   AssertionError: POST /model/new: 500 {"error":{"message":"Model create was saved to the database, but the model id(s) [...] are not live in this pod's router after the reload and are not being served by this pod. ..."}}   (sail/ deployment cannot be registered at base)
FAILED tests/integration/pricing/test_service_tier_pricing.py::test_balanced_service_tier_bills_balanced_rates_and_keeps_pricing_off_the_wire
FAILED tests/integration/pricing/test_service_tier_pricing.py::test_sail_flex_sends_completion_window_without_service_tier_and_bills_flex_rates
FAILED tests/integration/pricing/test_responses_extra_body_pricing.py::test_responses_extra_body_completion_window_on_a_sail_deployment_bills_the_window
3 failed, 5 passed, 1 skipped in 6.68s
```

The five tests that pass at base are behaviours the PR preserves rather than introduces (base-rate fallback, openai
shallow extra_body merge, Responses extra_body on the wire). Each is pinned by a named mutation below

## Mutations

Each mutation was applied to one production file in the tip worktree with an exact single-occurrence replacement,
the tip proxy was restarted, the test ran red, the original file was copied back, the proxy was restarted and the
test ran green. Ineffective mutations are listed too and are not counted as kills

| # | Production mutation | Test | Red run |
| --- | --- | --- | --- |
| 1 | `litellm/litellm_core_utils/llm_cost_calc/utils.py`: removed the `ServiceTier.BALANCED.value: ServiceTier.BALANCED.value,` entry from the service tier suffix mapping, so `_balanced` keys are never selected | test_balanced_service_tier_bills_balanced_rates_and_keeps_pricing_off_the_wire | `assert 0.06 == 0.22 ± 2.2e-07` (test_service_tier_pricing.py:53), `1 failed` |
| 2 | `litellm/litellm_core_utils/llm_cost_calc/utils.py`: `fallback_cost = model_info.get(base_key)` -> `fallback_cost = None`, breaking the tier-to-base rate fallback | test_balanced_and_flex_on_a_base_rate_only_model_bill_base_rates_and_pass_the_tier_through | `KeyError: 'x-litellm-response-cost'` (the proxy no longer priced the call), `1 failed` |
| 3 | `litellm/llms/openai/openai.py`: the SDK request options were built without `extra_body=data.get("extra_body")`, so extra_body never reached the wire | test_extra_body_metadata_on_an_openai_shaped_deployment_replaces_the_wire_metadata_whole[openai] and [cognition] | `Right contains 1 more item: {'metadata': {'b': 3, 'c': 4}}`, `2 failed, 4 deselected` |
| 4 | `litellm/cost_calculator.py`: dropped `"asap"` from the accepted completion windows in `_completion_window_value`, so an explicit `asap` no longer overrides `service_tier=flex` | test_sail_flex_sends_completion_window_without_service_tier_and_bills_flex_rates | `assert 0.013999999999999999 == 0.06 ± 6.0e-08`, `1 failed, 5 deselected` |
| 5 | `litellm/llms/openai_like/dynamic_config.py`: `key: value for key, value in body.items() if key != "service_tier"` -> `key: value for key, value in body.items()`, leaving `service_tier` on the Sail wire | test_sail_flex_sends_completion_window_without_service_tier_and_bills_flex_rates | `Left contains 1 more item: {'service_tier': 'flex'}`, `1 failed, 5 deselected` |
| 6 | `litellm/llms/base_llm/responses/transformation.py` `BaseResponsesAPIConfig.merge_extra_body`: `{**request, **extra_body}` -> `{**request, **{k: v for k, v in extra_body.items() if k != "metadata"}}`, dropping metadata from the Responses wire body | test_responses_extra_body_on_an_openai_shaped_deployment_reaches_the_wire_and_bills_base_rates | `Right contains 1 more item: {'metadata': {'completion_window': 'flex', 'trace': 'extra-body-control'}}` (test_responses_extra_body_pricing.py:78), `1 failed, 1 passed` |
| 7 | `litellm/responses/main.py` `responses()`: removed `**({"extra_body": extra_body} if extra_body else {})` from the `optional_params` passed to `update_from_kwargs`, so extra_body no longer reaches Responses logging | test_responses_extra_body_completion_window_on_a_sail_deployment_bills_the_window | `assert 0.2 == 0.05 ± 5.0e-08` (test_responses_extra_body_pricing.py:72, Sail flex window billed base), `1 failed, 1 passed` |
| 8 | `litellm/cost_calculator.py` `_provider_bills_by_completion_window`: `return provider is not None and provider.special_handling.get("service_tier_as_completion_window") is True` -> `return True`, billing every provider by `completion_window` | test_responses_extra_body_on_an_openai_shaped_deployment_reaches_the_wire_and_bills_base_rates | `assert 0.05 == 0.2 ± 2.0e-07` (test_responses_extra_body_pricing.py:72, openai deployment billed flex), `1 failed, 1 passed` |

Ineffective mutation, not counted: `litellm/llms/base_llm/chat/transformation.py` `BaseConfig.merge_extra_body`
made to drop `metadata`. The openai and cognition chat deployments both go through the OpenAI SDK path in
`litellm/llms/openai/openai.py`, which merges extra_body itself, so the observed body was unchanged and the metadata
test stayed green. Restored. Mutation 3 targets the path those deployments actually use

Kill count: 8 of 8 counted mutations killed (9 attempted, 1 discarded as not exercising the tested path)

Restore evidence: after every restore `git status --short` in the tip worktree listed only the four test files
(`tests/integration/_support/upstream.py`, `tests/integration/pricing/test_service_tier_pricing.py`,
`tests/integration/pricing/test_responses_extra_body_pricing.py`,
`tests/integration/providers/test_sail_audio_transcription.py`) plus this file, and the tip run above is the green
run on the fully restored tree

## Blockers

(6) `model_parameters.extra_body` in the spend row: not observable at the tip. `StandardLoggingPayload` carries
`model_parameters`, but `litellm/proxy/spend_tracking/spend_event.py` drops it before the row is written
(`_STANDARD_LOGGING_DROPPED_KEYS = frozenset({"model_parameters"})`) and `LiteLLM_SpendLogs` has no such column. The
row's `metadata` JSON for a `/v1/responses` call was inspected directly in Postgres and contains no `extra_body`. The
"once, not twice" claim therefore cannot be asserted from the spend row. What the test pins instead is the effect of
extra_body reaching logging optional_params: the Sail deployment bills `_flex` from `extra_body.metadata.completion_window`
and drops to base on `asap` (mutation 7 removes exactly the PR's `extra_body` logging hunk and the test goes red),
while the openai deployment with the same window bills base (mutation 8). The wire body is asserted whole for both

(7) `/v1/audio/transcriptions` on `sail/x`: the tip answers 500, not 4xx. Full observed body:
`{"error":{"message":"litellm.APIConnectionError: APIConnectionError: SailException - Unmapped provider passed in. Unable to get the response.\n\nLiteLLM: model group '<model>' failed with the error above. No fallback was attempted.","type":null,"param":null,"code":"500"}}`.
The scripted upstream observed zero requests for the call (the exception is raised in `litellm.main.atranscription`
before any HTTP call, confirmed from the proxy log traceback and an empty `/__observations` drain afterwards). Per
tests/integration/AGENTS.md the test keeps its intended 4xx and full-message assertions and is
`pytest.skip("BUG: ...")` at the top of the body. It was not mutation tested since it does not run. At base the same
test fails earlier because a `sail/` deployment cannot be registered at all (see the base run)

No real provider key was needed: every upstream in this cell is the scripted harness upstream, so the 1Password Sail
key was neither read nor used

## Gates

Run from the tip worktree with `/home/ubuntu/repos/litellm/.venv/bin` on PATH, `FILES` being the four touched test
files:

```
ruff check $FILES
All checks passed!
ruff format --check $FILES
4 files already formatted
basedpyright --pythonpath /home/ubuntu/repos/litellm/.venv/bin/python $FILES
```

basedpyright is not configured for `tests/integration` (`pyrightconfig.json` includes only `litellm`), so it reports
the directory's pre-existing patterns: `float(rows[0]["spend"])` on a `JsonValue` and `pytest.approx` being partially
unknown, both present in every neighbouring pricing test, plus errors in the untouched parts of `upstream.py`. The
lines this branch adds introduce no other category; the two it did introduce (`response.json()` returning `Any` and a
`Mapping` passed where a `dict` is expected) were fixed before the commit

```
python scripts/type_discipline_gate.py --base d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62
OK: every LIT rule is within its codebase ceiling (base d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62)
python scripts/ruff_strict_gate.py --base d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62
OK: every strict rule is within its codebase ceiling (base d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62)
```

## Owning groups through tests/integration/run.py

Both groups ran against the tip stack (`INTEGRATION_PROXY_URL=http://127.0.0.1:4000`,
`INTEGRATION_UPSTREAM_URL=http://127.0.0.1:8190`, `DATABASE_URL=...integration_tip`, `REDIS_HOST=127.0.0.1`,
`LITELLM_LOCAL_MODEL_COST_MAP=True`). `run.py` starts its own owned proxies for the tests that need one

```
python tests/integration/run.py accounting --results /home/ubuntu/sail-int/results/accounting2
============= 2 failed, 54 passed, 3 warnings in 721.92s (0:12:01) =============
FAILED tests/integration/spend/test_cache_and_quota.py::test_scheduled_budget_reset_reconnects_after_db_transport_failure_and_unblocks_key
FAILED tests/integration/spend/test_cache_and_quota.py::test_in_flight_count_tokens_does_not_reserve_key_budget_away_from_a_completion
```

```
python tests/integration/run.py providers --results /home/ubuntu/sail-int/results/providers
====== 2 failed, 149 passed, 1 skipped, 103 warnings in 365.32s (0:06:05) ======
FAILED tests/integration/providers/test_databricks_oauth_wire.py::test_databricks_ai_gateway_api_base_requests_oauth_token_from_workspace_origin
FAILED tests/integration/routing/test_redis_recovery.py::test_owned_redis_outage_recovers_requests_and_real_response_cache
```

Every test this branch adds passed in its group (the audio test is the documented `BUG:` skip). The four failures are
in files this branch does not touch and none of them reaches the scripted scenario route that the `upstream.py` change
covers: two are owned-proxy and owned-Redis lifecycle tests (`Owned proxy exited before readiness`, owned Redis
outage), one is the Gemini `countTokens` wire peer, one is the Databricks OAuth wire peer. The two accounting failures
were re-run with the pristine tip `upstream.py` (the file checked out from d7bc17fa, upstream restarted) and failed
the same way (`2 failed in 42.02s`), so they are independent of this branch

The first accounting run (before the observation scoping) also failed
`test_repeated_count_tokens_on_budgeted_key_does_not_reserve_budget_or_block_later_completion`: the scripted
catch-all recorded the proxy's `POST /v1/responses/input_tokens` probe, which that test asserts the upstream never
sees. The observation in `Provider.scripted` is therefore recorded only after the request resolves to a registered
scenario, which is the only case the new tests read. That test passed in the second run
