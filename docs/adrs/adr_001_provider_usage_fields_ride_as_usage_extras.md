# ADR 001: Provider usage fields ride as extras on the normalized usage object

Status: Accepted

## Context

Providers report usage fields that no other provider has, such as xAI's `usage.server_side_tool_usage_details.web_search_calls`, and cost tracking has to read them or that spend never reaches `LiteLLM_SpendLogs`. One provider response can be served on `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`, streaming or not, so a field that only one of those paths carries is billed wrong on the others

In [#30817](https://github.com/BerriAI/litellm/pull/30817) xAI web searches were not billed. The first fix carried the new field by reshaping usage inside the xAI Responses transform. That billed correctly, but `/v1/responses` callers then got chat-shaped usage, and the next provider or endpoint would have needed its own copy of the override. A generic path already existed and was not found

## Decision

Provider-only usage fields stay on the usage object as pydantic extras, and that provider's cost calculator reads them. No transform renames, removes, or reshapes the usage fields an endpoint's schema defines in order to feed cost tracking. A transform may fill a field LiteLLM's usage object already has, which is how the existing gates fire today: the xAI and Perplexity chat transforms set `prompt_tokens_details.web_search_requests`, the Groq chat transform sets `server_tool_use`, and both xAI transforms set `cost` from xAI's own `cost_in_usd_ticks`

`Usage` (`litellm/types/utils.py`) keeps every keyword argument it does not define as an attribute. On a non-streamed `/v1/chat/completions` response, a transform that hands the response to `convert_to_model_response_object` (`litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py`), as the OpenAI-compatible ones do, passes the provider's whole usage block, so its unknown keys survive. A transform that builds `Usage` from named fields, as the Anthropic, Gemini, and Bedrock Converse transforms do, keeps only the fields it names. `ResponseAPIUsage` (`litellm/types/llms/openai.py`) inherits `extra="allow"` from `BaseLiteLLMOpenAIResponseObject`, so unknown fields survive validation on the Responses side too. `ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage` (`litellm/responses/utils.py`) is the one bridge from Responses usage to chat `Usage`, and it copies `model_extra` across, minus a fixed exclusion list of chat and detail field names. Cost tracking, logging, and the completions-to-Responses bridge all call it

Two paths do not carry extras yet. Streamed chat rebuilds usage from a fixed list of fields in `ChunkProcessor.calculate_usage` (`litellm/litellm_core_utils/streaming_chunk_builder_utils.py`), and `/v1/responses` served by a chat-only provider builds `ResponseAPIUsage` from the token counts and a few named fields in `transform_chat_completion_usage_to_responses_usage` (`litellm/responses/litellm_completion_transformation/transformation.py`). A field that has to be billed on one of those paths is carried there as an extra too, never through a side channel

Built-in tool cost is gated in `StandardBuiltInToolCostTracking.get_cost_for_built_in_tools` (`litellm/litellm_core_utils/llm_cost_calc/tool_call_cost_tracking.py`), which gets the provider's rate through `get_cost_for_web_search_request` (`litellm/llms/__init__.py`). Per-provider arithmetic stays in `litellm/llms/<provider>/cost_calculator.py`

To bill a new provider usage field, read it in that provider's cost calculator, and when that provider's transform builds `Usage` from named fields, pass the new field there as well. When the field counts a built-in tool such as web search, add the provider to the `get_cost_for_web_search_request` dispatch, and add a gate to the shared tracker only when no existing gate fires. Then check that the field reaches the calculator on every path the proof covers, streaming included

## Alternatives Considered

**Reshape usage in the provider transform**, as #30817 first did. It changes the response schema that callers of that endpoint rely on, and every provider and endpoint needs its own copy

**A typed field on `ResponseAPIUsage` and `Usage` for each provider's quirk**. The types grow without bound and the bridge still needs an update for each one. A field that several providers report, like `server_tool_use`, does become a typed field

**A side channel such as `_hidden_params`**. Cost calculators receive `Usage`, so a side channel makes a second source of truth for the same number

**Provider branches inside the shared tracker**. It puts one provider's pricing in a file that every provider shares, which #30817 reverted

## Consequences

Benefits:

- Billing a new provider usage field touches that provider's calculator plus at most one gate and one dispatch arm, and it works on every path that carries extras

Costs:

- Extras are untyped, so a reader validates them before use, as `_usage_reports_server_side_web_search_calls` does
- The bridge drops an extra whose name is on its exclusion list (`prompt_tokens`, `total_tokens`, `input_token_details`, and the rest), so a new extra with one of those names needs that list checked
- Streamed chat and chat-to-Responses usage drop extras until those paths carry them, so a field billed there needs that path extended, and tests and proof runs cover `"stream": true` too
- This covers response usage only. Request-side provider params (`get_supported_openai_params`, `map_openai_params`) are a separate mechanism
