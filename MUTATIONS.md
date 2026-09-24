# Mutation evidence for the tests/e2e Sail service_tier suites (PR #42840)

Review evidence for branch `litellm_sail_tests_e2e`, created from tip d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 (merge base 0c1c3e18d5250ec3a0e1e3f287e0b93e2906d900). This file is not meant to merge

## Tests on the branch

| Test | File |
| --- | --- |
| `TestSailServiceTier::test_chat_prices_each_service_tier_as_its_window_and_echoes_none` | tests/e2e/llm_translation/test_sail_service_tier_e2e.py |
| `TestSailServiceTier::test_responses_extra_body_completion_window_prices_that_window` | tests/e2e/llm_translation/test_sail_service_tier_e2e.py |
| `TestSailServiceTier::test_flex_only_model_serves_flex_and_refuses_the_windows_other_tiers_become` | tests/e2e/llm_translation/test_sail_service_tier_e2e.py |
| `TestSailCompletionWindowPricing::test_each_service_tier_bills_its_windows_cost_map_rates` | tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py |
| `TestSailCompletionWindowPricing::test_responses_completion_window_in_extra_body_bills_balanced_rates` | tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py |
| `TestSailCompletionWindowPricing::test_flex_only_model_refuses_default_tier_and_bills_nothing` | tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py |
| `TestOpenAIKeepsItsOwnServiceTier::test_openai_bills_the_tier_it_reports_and_refuses_balanced_unbilled` | tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py |

Shared helpers live in tests/e2e/completion_window_pricing.py (runtime cost-map lookup of the cheapest Sail model that carries base, `_balanced` and `_flex` rates, and of the cheapest flex-only Sail model, plus the 1e-9 relative tolerance helper). tests/e2e/models.py gained the six `*_balanced` and `*_flex` cost-map fields on `CostMapEntry`

## Harness

Proxy booted per tests/e2e/CONTRIBUTING.md from the repo checkout with `LITELLM_LOCAL_MODEL_COST_MAP=True`, real Postgres (`DATABASE_URL`), real Redis, `proxy_batch_write_at: 5`, and `SAIL_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` from the 1Password Shared vault in the proxy environment. Every mutation run was: edit one production file, restart the proxy, run the tests named below with `--reruns 0`, restore the file, confirm `git status` is clean under `litellm/`. No test file was changed between the red and the green run of a mutation. The mutation runs happened before the final refactor that swapped `pytest.approx` for the typed `within_rel` / `all_within_rel` helpers; the assertion messages are identical, only the trailing `assert` explanation pytest prints differs. The final green run below is on the exact branch contents

At the merge base 0c1c3e18d5250ec3a0e1e3f287e0b93e2906d900 there is no `sail` provider, so every Sail test fails at `/model/new` before making a request. The mutations below are the meaningful bar: each one leaves the provider registered and breaks one behaviour the PR introduces

## Kill table: 5 killed / 5 applied

| Id | File | Mutation | Tests that went red |
| --- | --- | --- | --- |
| M1 | litellm/llms/openai_like/dynamic_config.py | `ServiceTier.BALANCED.value: "balanced"` -> `ServiceTier.BALANCED.value: "flex"` | `test_flex_only_model_serves_flex_and_refuses_the_windows_other_tiers_become` |
| M2 | litellm/cost_calculator.py | `_provider_bills_by_completion_window` body -> `return False` | `test_responses_extra_body_completion_window_prices_that_window`, `test_responses_completion_window_in_extra_body_bills_balanced_rates` |
| M3 | litellm/litellm_core_utils/llm_cost_calc/utils.py | removed `ServiceTier.BALANCED.value: ServiceTier.BALANCED.value,` from `_SERVICE_TIER_TO_COST_KEY_SUFFIX` | `test_chat_prices_each_service_tier_as_its_window_and_echoes_none`, `test_each_service_tier_bills_its_windows_cost_map_rates` |
| M4 | litellm/llms/openai_like/dynamic_config.py | `"default": "asap"` -> `"default": "flex"` | `test_flex_only_model_serves_flex_and_refuses_the_windows_other_tiers_become`, `test_flex_only_model_refuses_default_tier_and_bills_nothing` |
| M5 | litellm/cost_calculator.py | non-window-provider branch of `_service_tier_billed_by_completion_window`: `return service_tier` -> `return ServiceTier.FLEX.value` | `test_openai_bills_the_tier_it_reports_and_refuses_balanced_unbilled` |

Every test on the branch is red under at least one of M1 to M5

## M1 balanced window sent as flex

```diff
@@ -21,7 +21,7 @@ from .json_loader import SimpleProviderConfig
-        ServiceTier.BALANCED.value: "balanced",
+        ServiceTier.BALANCED.value: "flex",
```

Run: `pytest tests/e2e/llm_translation/test_sail_service_tier_e2e.py tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py -k "flex_only or each_service_tier_bills"`

Result: `1 failed, 2 passed, 2 deselected in 171.37s`

Failing assertion (test_sail_service_tier_e2e.py:217):

```
AssertionError: expected sail's 400 for service_tier='balanced', got 200: {"id":"resp_01a0d42b-225c-7fb0-8265-71b0f6285c28",...,"model":"sail-flex-6bb4ace67ce7","object":"chat.completion",...,"usage":{"completion_tokens":64,"prompt_tokens":28,"total_tokens":92,...}}
assert 200 == 400
```

The flex-only Sail model accepts `completion_window=flex` and refuses `balanced`; with balanced rewritten to flex the provider served the call instead of refusing it. The two billing tests kept passing under this mutation because balanced and flex both still price through the cost-map suffix path, which is why the flex-only refusal test exists

Restore: `git status --short -- litellm/llms/openai_like/dynamic_config.py` empty

## M2 completion_window billing gate off

```diff
@@ -982,7 +982,7 @@ def _provider_bills_by_completion_window(custom_llm_provider: str | None) -> boo
-    return provider is not None and provider.special_handling.get("service_tier_as_completion_window") is True
+    return False
```

Run: `pytest ... -k responses`

Result: `2 failed, 5 deselected in 107.53s`

Failing assertions:

```
test_sail_service_tier_e2e.py:177: AssertionError: resp_QeexFiEG... priced 3.96e-06, expected 3.08e-06 from input_tokens=16 output_tokens=14 input_tokens_details=_ResponsesInputDetails(cached_tokens=0) output_tokens_details=_ResponsesOutputDetails(reasoning_tokens=12) at the balanced rates TierRates(input=7e-08, output=1.4e-07, cache_read=2e-08)
```

```
test_sail_completion_window_pricing_e2e.py:157: AssertionError: balanced row resp_vHYEtxB9... billed (input, output, cache_read, reasoning, total) (1.8e-06, 8.1e-06, 0.0, 7.74e-06, 9.9e-06), expected (1.4000000000000001e-06, 6.300000000000001e-06, 0.0, 6.020000000000001e-06, 7.7e-06) from prompt_tokens=20 cached_tokens=0 completion_tokens=45 reasoning_tokens=43 at the balanced rates TierRates(input=7e-08, output=1.4e-07, cache_read=2e-08)
```

Both `/v1/responses` calls were billed at the base (asap) rates once the gate stopped recognising Sail's `completion_window`

Restore: the mutate script's own restore step tripped on its text-match guard, so the line was restored by hand and `git status --short -- litellm/cost_calculator.py` confirmed empty before the next run

## M3 balanced cost suffix dropped

```diff
@@ -69,7 +69,6 @@ _SERVICE_TIER_SUFFIXES: Final[tuple[str, ...]] = tuple(
-        ServiceTier.BALANCED.value: ServiceTier.BALANCED.value,
```

Run: `pytest ... -k "each_service_tier or chat_prices_each"`

Result: `2 failed in 130.02s`

Failing assertions:

```
test_sail_service_tier_e2e.py:146: AssertionError: x-litellm-response-cost per tier {'flex': 4.78e-06, 'balanced': 7.92e-06, None: 8.64e-06} != each window's rates times its usage {'flex': 4.78e-06, 'balanced': 6.16e-06, None: 8.64e-06} (rates WindowPricedModel(model='sail/deepseek-ai/DeepSeek-V4-Flash-0731', asap=TierRates(input=9e-08, output=1.8e-07, cache_read=2e-08), balanced=TierRates(input=7e-08, output=1.4e-07, cache_read=2e-08), flex=TierRates(input=5e-08, output=9e-08, cache_read=1e-08)))
```

```
test_sail_completion_window_pricing_e2e.py:157: AssertionError: balanced row resp_01a0d432-5695-7a4b-93d6-1aaf5f91188f billed (input, output, cache_read, reasoning, total) (1.62e-06, 9.36e-06, 0.0, 9e-06, 1.098e-05), expected (1.26e-06, 7.280000000000001e-06, 0.0, 7.000000000000001e-06, 8.540000000000001e-06) from prompt_tokens=18 cached_tokens=0 completion_tokens=52 reasoning_tokens=50 at the balanced rates TierRates(input=7e-08, output=1.4e-07, cache_read=2e-08)
```

Balanced chat calls fell back to base rates; flex and absent stayed correct, so exactly the balanced cell moved

Restore: restored by hand after the script's guard tripped, `git status` clean under `litellm/`

## M4 default tier sent as flex

```diff
@@ -23,7 +23,7 @@ _SERVICE_TIER_TO_COMPLETION_WINDOW: Final[Mapping[str, str]] = MappingProxyType(
-        "default": "asap",
+        "default": "flex",
```

Run: `pytest ... -k flex_only`

Result: `2 failed in 160.92s`

Failing assertions:

```
test_sail_service_tier_e2e.py:217: AssertionError: expected sail's 400 for service_tier='default', got 200: {"id":"resp_01a0d435-10b7-75b2-9bdf-7512013c8494",...,"model":"sail-flex-e9a7dbaa3bc9",...}
assert 200 == 400
```

```
test_sail_completion_window_pricing_e2e.py:292: AssertionError: expected the provider's 400, got 200: {"id":"resp_01a0d436-865b-774b-a4fe-b08fe6e2e592",...,"model":"sail-flexonly-656400d84243",...}
assert 200 == 400
```

Restore: `git status --short -- litellm/llms/openai_like/dynamic_config.py` empty

## M5 window billing hook leaks flex onto other providers

```diff
@@ -999,7 +999,7 @@ def _service_tier_billed_by_completion_window(
-        return service_tier
+        return ServiceTier.FLEX.value
```

Run: `pytest tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py -k openai`

Result: `1 failed in 88.49s`

Failing assertion:

```
test_sail_completion_window_pricing_e2e.py:157: AssertionError: openai default row chatcmpl-ERgTzeELuKMNg9PbdAIuPhXmJDUBO billed (input, output, cache_read, reasoning, total) (4.75e-05, 0.000255, 0.0, 0.000105, 0.00030250000000000003), expected (9.5e-05, 0.00051, 0.0, 0.00021, 0.0006050000000000001) from prompt_tokens=19 cached_tokens=0 completion_tokens=17 reasoning_tokens=7 at the openai default rates TierRates(input=5e-06, output=3e-05, cache_read=5e-07)
```

The untiered OpenAI row was priced at OpenAI's `_flex` cost-map rates and its `cost_breakdown.service_tier` read `flex`, so the Sail-only hook had leaked into every provider's billing. The Sail tests all stay green under M5, which is the point of keeping the OpenAI test

Restore: `git status --short -- litellm/cost_calculator.py` empty

## Final green run at the tip, production code unmutated

`git status --short -- litellm/` empty, proxy restarted, then

```
LITELLM_PROXY_URL=http://localhost:4000 LITELLM_MASTER_KEY=sk-1234 .venv/bin/python -m pytest tests/e2e/llm_translation/test_sail_service_tier_e2e.py tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py --reruns 0 -p no:cacheprovider -v -rA --tb=short
```

```
PASSED tests/e2e/llm_translation/test_sail_service_tier_e2e.py::TestSailServiceTier::test_chat_prices_each_service_tier_as_its_window_and_echoes_none
PASSED tests/e2e/llm_translation/test_sail_service_tier_e2e.py::TestSailServiceTier::test_responses_extra_body_completion_window_prices_that_window
PASSED tests/e2e/llm_translation/test_sail_service_tier_e2e.py::TestSailServiceTier::test_flex_only_model_serves_flex_and_refuses_the_windows_other_tiers_become
PASSED tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py::TestSailCompletionWindowPricing::test_each_service_tier_bills_its_windows_cost_map_rates
PASSED tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py::TestSailCompletionWindowPricing::test_responses_completion_window_in_extra_body_bills_balanced_rates
PASSED tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py::TestSailCompletionWindowPricing::test_flex_only_model_refuses_default_tier_and_bills_nothing
PASSED tests/e2e/quota_management/spend_tracking/test_sail_completion_window_pricing_e2e.py::TestOpenAIKeepsItsOwnServiceTier::test_openai_bills_the_tier_it_reports_and_refuses_balanced_unbilled
======================== 7 passed in 264.95s (0:04:24) =========================
```

## Cells the live providers would not let the tests prove

OpenAI `service_tier: balanced` with the latest `openai/gpt-5.5` row on the account behind the vault key is refused by OpenAI itself: `Invalid value: 'balanced'. Supported values are: 'auto', 'default', 'fast', 'flex', and 'priority'.` The test therefore proves the tier is forwarded untouched (OpenAI's own 400 comes back, with a zero-spend failure row and no cost_breakdown) and that the untiered call is billed at the tier OpenAI reports back. The requested cell "openai balanced bills the same as no tier" is UNVERIFIED: no OpenAI model on this account accepts `balanced`

Anthropic `service_tier: balanced` versus no tier is UNVERIFIED: every request through the vault's Anthropic key fails with `Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.`, so no Anthropic test is on the branch rather than a test that skips or passes vacuously

Sail `/v1/responses` returns `metadata.completion_window: "standard"` alongside `supercache_*` fields even when the request carried `balanced`, so the tests do not pin Sail's echoed metadata (a vendor fact) and assert the balanced billing instead, which M2 shows is driven by the requested window
