# Mutation evidence for litellm_sail_tests_unit

Tests-only regression branch for PR 42840, created from tip d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 (merge base 0c1c3e18d5250ec3a0e1e3f287e0b93e2906d900). Every new test was run three ways: at the merge base with the changed test files copied in (red unless noted below), at the tip with one named production mutation applied (red), then at the tip with the mutated file restored from d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 (green). Production files were diffed against d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 before every mutation and after every restore, and the final `git status` of the branch shows test files plus this document only

## Merge-base run

Command, run in a detached worktree of the merge base with the eleven changed test files and the new chat transformation test copied in:

```
LITELLM_LOCAL_MODEL_COST_MAP=True pytest -p no:cacheprovider <the tests listed in the PR report> -q
```

Result: 15 failed, 13 passed. The 13 that pass at the merge base are the 12 parametrized `test_completion_cost_balanced_tier_bills_standard_rates_on_rows_without_balanced_keys` cases (the balanced tier does not exist there, so it takes the unknown-tier path and trivially equals the no-tier cost; M01 and M02 are their kill evidence) and `test_completion_cost_ignores_completion_window_for_openai` (the merge base has no completion-window gate at all, so it bills base rates for every provider; M15 is its kill evidence). The 15 failures are import errors for `merge_extra_body`, `KNOWN_REQUEST_SERVICE_TIERS` and `special_handling`, the missing `sail` slug and `_balanced` keys, `sail/` resolving to `LLM Provider NOT provided`, the transcription call routing to OpenAI instead of failing before HTTP, and `extra_body` missing from the Responses logging optional_params

## Named production mutations

Each mutation ran `LITELLM_LOCAL_MODEL_COST_MAP=True pytest -p no:cacheprovider <tests> -q` at the tip, once mutated and once restored. Kill count: 15 of 15

### M01 area 1/3: drop the service-tier suffix fallback in _get_cost_per_unit

File: `litellm/litellm_core_utils/llm_cost_calc/utils.py`

Tests:

```
tests/test_litellm/test_cost_calculator.py::test_completion_cost_balanced_tier_bills_standard_rates_on_rows_without_balanced_keys
tests/test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_get_cost_per_unit_falls_back_from_balanced_key_to_base
```

Mutation (exact string replace):

```
    if cost_per_unit is None:
        # Check if any service tier suffix exists in the cost key
        for suffix in _SERVICE_TIER_SUFFIXES:
```

with:

```
    if cost_per_unit is None:
        # Check if any service tier suffix exists in the cost key
        for suffix in ():
```

Red (mutated), killed: true

```
E       AssertionError: (model, no tier, balanced, unknown tier) rows that disagree: (('claude-fable-5', 0.0173, 0.0, 0.0173), ('claude-fable-5-1', 0.017075, 0.0, 0.017075), ('claude-haiku-4-5', 0.00173, 0.0, 0.00173), ('claude-haiku-4-5-20251001', 0.00173, 0.0, 0.00173), ('claude-mythos-5', 0.0173, ...
E       assert (('claude-fab....017075), ...) == ()
E         
E         Left contains 20 more items, first extra item: ('claude-fable-5', 0.0173, 0.0, 0.0173)
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/litellm_core_utils/llm_cost_calc/utils.py`, then the same command

Green (restored), exit 0: `13 passed in 1.64s`

### M02 area 1/3/4: remove BALANCED from the ServiceTier enum

File: `litellm/types/utils.py`

Tests:

```
tests/test_litellm/integrations/test_prometheus_service_tier_label.py::test_requested_tier_recognizes_balanced_and_drops_garbage_without_touching_the_other_tiers
tests/test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_threshold_keys_exclude_balanced_variants_and_still_parse_k_thresholds
```

Mutation (exact string replace):

```
    AUTO = "auto"
    BALANCED = "balanced"
    FLEX = "flex"
```

with:

```
    AUTO = "auto"
    FLEX = "flex"
```

Red (mutated), killed: true

```
E   AttributeError: type object 'ServiceTier' has no attribute 'BALANCED'
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/types/utils.py`, then the same command

Green (restored), exit 0: `2 passed in 0.65s`

### M03 area 3: map the balanced tier onto the flex cost suffix

File: `litellm/litellm_core_utils/llm_cost_calc/utils.py`

Tests:

```
tests/test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_threshold_keys_exclude_balanced_variants_and_still_parse_k_thresholds
```

Mutation (exact string replace):

```
        ServiceTier.BALANCED.value: ServiceTier.BALANCED.value,
```

with:

```
        ServiceTier.BALANCED.value: ServiceTier.FLEX.value,
```

Red (mutated), killed: true

```
E       assert (4e-06, 6e-06) == (7e-06, 1.4e-05)
E         
E         At index 0 diff: 4e-06 != 7e-06
E         Use -v to get more diff
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/litellm_core_utils/llm_cost_calc/utils.py`, then the same command

Green (restored), exit 0: `1 passed in 0.42s`

### M04 area 2: put a balanced key on a non-sail cost-map row

File: `model_prices_and_context_window.json`

Tests:

```
tests/test_litellm/test_model_prices_schema.py::test_balanced_tier_keys_live_only_on_sail_rows_next_to_their_base_key
```

Mutation (regex replace):

```
("gpt-4o-mini": \{\n)
```

with:

```
\1        "input_cost_per_token_balanced": 1e-07,\n
```

Red (mutated), killed: true

```
E       AssertionError: (model, balanced key, litellm_provider, has base key) for balanced tier pricing that is off a sail/ row or missing its base key: (('gpt-4o-mini', 'input_cost_per_token_balanced', 'openai', True),)
E       assert (('gpt-4o-min...enai', True),) == ()
E         
E         Left contains one more item: ('gpt-4o-mini', 'input_cost_per_token_balanced', 'openai', True)
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- model_prices_and_context_window.json`, then the same command

Green (restored), exit 0: `1 passed in 0.40s`

### M05 area 5: BaseConfig.merge_extra_body deep-merges nested metadata

File: `litellm/llms/base_llm/chat/transformation.py`

Tests:

```
tests/test_litellm/llms/base_llm/chat/test_base_chat_transformation.py::test_default_merge_extra_body_shallow_merges_and_lets_extra_body_replace_nested_metadata
```

Mutation (exact string replace):

```
        return {**request, **extra_body} if extra_body else request  # mutable-ok: wire request body is a plain dict
```

with:

```
        if not extra_body:
            return request
        merged = {**request, **extra_body}
        if isinstance(request.get('metadata'), dict) and isinstance(extra_body.get('metadata'), dict):
            merged['metadata'] = {**request['metadata'], **extra_body['metadata']}
        return merged
```

Red (mutated), killed: true

```
E       AssertionError: assert {'messages': ...re': 0.2, ...} == {'messages': ...re': 0.2, ...}
E         
E         Omitting 4 identical items, use -vv to show
E         Differing items:
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/llms/base_llm/chat/transformation.py`, then the same command

Green (restored), exit 0: `1 passed in 0.41s`

### M06 area 5: BaseResponsesAPIConfig.merge_extra_body deep-merges nested metadata

File: `litellm/llms/base_llm/responses/transformation.py`

Tests:

```
tests/test_litellm/llms/base_llm/responses/test_transformation.py::test_default_merge_extra_body_shallow_merges_and_lets_extra_body_replace_nested_metadata
```

Mutation (exact string replace):

```
        return {**request, **extra_body} if extra_body else request  # mutable-ok: wire request body is a plain dict
```

with:

```
        if not extra_body:
            return request
        merged = {**request, **extra_body}
        if isinstance(request.get('metadata'), dict) and isinstance(extra_body.get('metadata'), dict):
            merged['metadata'] = {**request['metadata'], **extra_body['metadata']}
        return merged
```

Red (mutated), killed: true

```
E       AssertionError: assert {'input': 'hi...el': 'm', ...} == {'input': 'hi...el': 'm', ...}
E         
E         Omitting 4 identical items, use -vv to show
E         Differing items:
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/llms/base_llm/responses/transformation.py`, then the same command

Green (restored), exit 0: `1 passed in 0.39s`

### M07 area 5: llm_http_handler.completion inlines the shallow merge instead of calling the hook

File: `litellm/llms/custom_httpx/llm_http_handler.py`

Tests:

```
tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py::test_completion_merges_extra_body_through_the_real_config_hook
```

Mutation (exact string replace):

```
            data: Final = provider_config.merge_extra_body(transformed, extra_body)
```

with:

```
            data: Final = {**transformed, **extra_body} if extra_body else transformed
```

Red (mutated), killed: true

```
E       AssertionError: assert {'messages': ...re': 0.2, ...} == {'messages': ...d': 'x'}, ...}
E         
E         Omitting 3 identical items, use -vv to show
E         Differing items:
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/llms/custom_httpx/llm_http_handler.py`, then the same command

Green (restored), exit 0: `1 passed in 0.43s`

### M08 area 5: llm_http_handler response_api_handler (sync + async) inlines the shallow merge instead of calling the hook

File: `litellm/llms/custom_httpx/llm_http_handler.py`

Tests:

```
tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py::test_response_api_handler_merges_extra_body_through_the_real_config_hook
tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py::test_async_response_api_handler_merges_extra_body_through_the_real_config_hook
```

Mutation (exact string replace):

```
        data = responses_api_provider_config.merge_extra_body(data, extra_body)
```

with:

```
        data = {**data, **extra_body} if extra_body else data
```

Red (mutated), killed: true

```
E       AssertionError: assert {'input': 'hi...el': 'm', ...} == {'input': 'hi...el': 'm', ...}
E         
E         Omitting 3 identical items, use -vv to show
E         Differing items:
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/llms/custom_httpx/llm_http_handler.py`, then the same command

Green (restored), exit 0: `2 passed in 0.45s`

### M09 area 6: responses/main.py stops passing extra_body into logging optional_params

File: `litellm/responses/main.py`

Tests:

```
tests/test_litellm/responses/test_responses_api_request_body.py::test_aresponses_logs_extra_body_once_in_optional_params_and_leaves_the_wire_body_alone
```

Mutation (exact string replace):

```
                **(
                    {"extra_body": extra_body} if extra_body else {}
                ),  # mutable-ok: update_from_kwargs stores a plain dict
```

with:

```

```

Red (mutated), killed: true

```
E       AssertionError: assert ({'max_output...warg': '0'}},) == ({'extra_body...warg': '0'}},)
E         
E         At index 0 diff: {'max_output_tokens': 50, 'metadata': {'from_kwarg': '0'}} != {'max_output_tokens': 50, 'metadata': {'from_kwarg': '0'}, 'extra_body': {'metadata': {'from_extra_body': '1'}, 'vendor_only_field': 'x'}}
E         Use -v to get more diff
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/responses/main.py`, then the same command

Green (restored), exit 0: `1 passed in 0.84s`

### M10 area 7: keep sail inside OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS

File: `litellm/constants.py`

Tests:

```
tests/test_litellm/test_constants.py::test_openai_transcription_providers_are_the_openai_compatible_set_minus_sail
tests/test_litellm/llms/openai_like/test_sail_provider.py::TestSailRequestShape::test_sail_transcription_rejected_without_hitting_sail
```

Mutation (exact string replace):

```
    frozenset(openai_compatible_providers) - frozenset(("sail",))
```

with:

```
    frozenset(openai_compatible_providers)
```

Red (mutated), killed: true

```
E       AssertionError: assert frozenset({'a...ure_ai', ...}) == frozenset({'a...ure_ai', ...})
E         
E         Extra items in the left set:
E         'sail'
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/constants.py`, then the same command

Green (restored), exit 0: `2 passed in 0.84s`

### M11 area 8: JSON loader drops the special_handling block of every provider

File: `litellm/llms/openai_like/json_loader.py`

Tests:

```
tests/test_litellm/llms/openai_like/test_json_providers.py::test_json_loader_yields_the_merge_base_params_and_special_handling_for_every_non_sail_slug
```

Mutation (exact string replace):

```
        self.special_handling: Mapping[str, object] = data.get("special_handling") or MappingProxyType({})
```

with:

```
        self.special_handling: Mapping[str, object] = MappingProxyType({})
```

Red (mutated), killed: true

```
E       AssertionError: assert {'abliteratio...', ...]}, ...} == {'abliteratio...', ...]}, ...}
E         
E         Omitting 26 identical items, use -vv to show
E         Differing items:
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/llms/openai_like/json_loader.py`, then the same command

Green (restored), exit 0: `1 passed in 7.49s`

### M12 area 8: dynamic_config drops temperature from every JSON provider's supported params

File: `litellm/llms/openai_like/dynamic_config.py`

Tests:

```
tests/test_litellm/llms/openai_like/test_json_providers.py::test_json_loader_yields_the_merge_base_params_and_special_handling_for_every_non_sail_slug
```

Mutation (exact string replace):

```
            excluded_params: Final = frozenset(tool_params) | frozenset(provider.unsupported_params)
```

with:

```
            excluded_params: Final = frozenset(tool_params) | frozenset(provider.unsupported_params) | {"temperature"}
```

Red (mutated), killed: true

```
E       AssertionError: assert {'abliteratio...', ...]}, ...} == {'abliteratio...', ...]}, ...}
E         
E         Differing items:
E         {'apertis': {'special_handling': {}, 'supported_openai_params': ['audio', 'extra_headers', 'frequency_penalty', 'logit_bias', 'logprobs', 'max_completion_tokens', ...]}} != {'apertis': {'special_handling': {}, 'supported_openai_params': ['audio', 'extra_headers', 'frequency_penalty', 'logi ...
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/llms/openai_like/dynamic_config.py`, then the same command

Green (restored), exit 0: `1 passed in 7.78s`

### M13 area 9: get_model_info never fills input_cost_per_token_balanced

File: `litellm/utils.py`

Tests:

```
tests/test_litellm/test_utils.py::test_get_model_info_reports_balanced_tier_prices_only_where_the_row_has_them
```

Mutation (exact string replace):

```
                input_cost_per_token_balanced=_model_info.get("input_cost_per_token_balanced", None),
```

with:

```
                input_cost_per_token_balanced=None,
```

Red (mutated), killed: true

```
E       assert (None, 2.5e-0...7, None, None) == (5e-07, 2.5e-...7, None, None)
E         
E         At index 0 diff: None != 5e-07
E         Use -v to get more diff
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/utils.py`, then the same command

Green (restored), exit 0: `1 passed, 6 warnings in 0.93s`

### M14 area 10: completion_cost runs the completion-window gate before inferring the provider

File: `litellm/cost_calculator.py`

Tests:

```
tests/test_litellm/test_cost_calculator.py::test_completion_cost_infers_sail_from_model_before_the_completion_window_gate
```

Mutation (exact string replace):

```
                if custom_llm_provider is None:
                    try:
                        model, custom_llm_provider, _, _ = litellm.get_llm_provider(
                            model=model
                        )  # strip the llm provider from the model name -> for image gen cost calculation
                    except Exception as e:
                        verbose_logger.debug(
                            "litellm.cost_calculator.py::completion_cost() - Error inferring custom_llm_provider - %s",
                            e,
                        )
                service_tier = _service_tier_billed_by_completion_window(
                    service_tier=service_tier,
                    optional_params=optional_params,
                    custom_llm_provider=custom_llm_provider,
                )
```

with:

```
                service_tier = _service_tier_billed_by_completion_window(
                    service_tier=service_tier,
                    optional_params=optional_params,
                    custom_llm_provider=custom_llm_provider,
                )
                if custom_llm_provider is None:
                    try:
                        model, custom_llm_provider, _, _ = litellm.get_llm_provider(
                            model=model
                        )  # strip the llm provider from the model name -> for image gen cost calculation
                    except Exception as e:
                        verbose_logger.debug(
                            "litellm.cost_calculator.py::completion_cost() - Error inferring custom_llm_provider - %s",
                            e,
                        )
```

Red (mutated), killed: true

```
E       assert 0.001596 == 0.00075999999...9999 ± 1.0e-12
E         
E         comparison failed
E         Obtained: 0.001596
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/cost_calculator.py`, then the same command

Green (restored), exit 0: `1 passed in 0.55s`

### M15 area 10: completion-window gate ignores the provider flag (every provider bills by completion_window)

File: `litellm/cost_calculator.py`

Tests:

```
tests/test_litellm/test_cost_calculator.py::test_completion_cost_ignores_completion_window_for_openai
```

Mutation (exact string replace):

```
    if optional_params is None or not _provider_bills_by_completion_window(custom_llm_provider):
```

with:

```
    if optional_params is None:
```

Red (mutated), killed: true

```
E       assert 0.01 == 0.02 ± 2.0e-11
E         
E         comparison failed
E         Obtained: 0.01
```

Restore: `git checkout d7bc17fa47ee9acacd441e6b3349c7aa9e1c6c62 -- litellm/cost_calculator.py`, then the same command

Green (restored), exit 0: `1 passed in 0.45s`
