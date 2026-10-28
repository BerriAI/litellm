# ADR 001: Provider usage fields ride as extras on the normalized usage object

Status: Accepted

## Context

Providers report usage fields that no other provider has, such as xAI's `usage.server_side_tool_usage_details.web_search_calls`, and cost tracking has to read them or that spend never reaches `LiteLLM_SpendLogs`. One provider response can be served on `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`, streaming or not, so a field that only one of those paths carries is billed wrong on the others

In [#30817](https://github.com/BerriAI/litellm/pull/30817) xAI web searches were not billed. The first fix carried the new field by reshaping usage inside the xAI Responses transform. That billed correctly, but `/v1/responses` callers then got chat-shaped usage, and the next provider or endpoint would have needed its own copy of the override. A generic path already existed and was not found

## Decision

Provider-only usage fields stay on the usage object as pydantic extras, and that provider's cost calculator reads them. No transform changes what the caller sees in order to feed cost tracking

`ResponseAPIUsage` (`litellm/types/llms/openai.py`) inherits `extra="allow"` from `BaseLiteLLMOpenAIResponseObject`, so unknown fields survive validation. `ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage` (`litellm/responses/utils.py`) is the one bridge from Responses usage to chat `Usage`, and it copies `model_extra` across, minus the keys it sets itself. Cost tracking, logging, and the completions-to-Responses bridge all call it

Built-in tool cost is gated in `StandardBuiltInToolCostTracking.get_cost_for_built_in_tools` (`litellm/litellm_core_utils/llm_cost_calc/tool_call_cost_tracking.py`), which gets the provider's rate through `get_cost_for_web_search_request` (`litellm/llms/__init__.py`). Per-provider arithmetic stays in `litellm/llms/<provider>/cost_calculator.py`

To bill a new provider usage field, read it in that provider's cost calculator, add the provider to the dispatch, and add a gate to the shared tracker only when no existing gate fires

## Alternatives Considered

**Reshape usage in the provider transform**, as #30817 first did. It changes the response schema that callers of that endpoint rely on, and every provider and endpoint needs its own copy

**A typed field on `ResponseAPIUsage` and `Usage` for each provider's quirk**. The types grow without bound and the bridge still needs an update for each one. A field that several providers report, like `server_tool_use`, does become a typed field

**A side channel such as `_hidden_params`**. Cost calculators receive `Usage`, so a side channel makes a second source of truth for the same number

**Provider branches inside the shared tracker**. It puts one provider's pricing in a file that every provider shares, which #30817 reverted

## Consequences

Benefits:

- Billing a new provider usage field touches that provider's calculator plus at most one gate and one dispatch arm, and it works on every endpoint that goes through the bridge

Costs:

- Extras are untyped, so a reader validates them before use, as `_usage_reports_server_side_web_search_calls` does
- The bridge drops an extra that has the name of a field it sets itself (`prompt_tokens`, `total_tokens`, and the others in its exclusion list), so a new extra with a chat usage name needs that list checked
- Streaming assembles usage on a different path, so tests and proof runs cover `"stream": true` too
- This covers response usage only. Request-side provider params (`get_supported_openai_params`, `map_openai_params`) are a separate mechanism
