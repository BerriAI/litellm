# 0001. Provider usage extras ride on the normalized Usage object

Status: Accepted

Date: 2026-08-12

## Context

Providers keep inventing usage fields. xAI reports server-side tool calls as `usage.server_side_tool_usage_details.web_search_calls`, Anthropic reports `usage.server_tool_use.web_search_requests`, Gemini hides search counts in `prompt_tokens_details`, and OpenAI reports nothing about a web search in usage at all. Cost tracking has to see those numbers, because a search call the gateway doesn't meter is spend that shows up on the provider invoice and nowhere in `LiteLLM_SpendLogs`

The same response can also be reached through three request surfaces (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) and a bridge that serves one surface from another provider's shape, so a usage field that only survives on one of those paths is a bug on the other two

The decision was forced by [#30817](https://github.com/BerriAI/litellm/pull/30817), where xAI web searches billed at $0. The first attempt carried the new field by overriding the xAI Responses transform to swap `response.usage` to the chat `Usage` shape. That billed correctly and broke the `/v1/responses` contract for every xAI caller, since clients then got `prompt_tokens` where the OpenAI Responses schema promises `input_tokens`, and it would have needed the same override again in the next provider and the next endpoint. The machinery to carry the field already existed and was not found

## Decision

Provider-specific usage fields travel as extra fields on the usage object and are read by provider cost calculators after normalization. Nothing on the wire is reshaped to make cost tracking work

Concretely:

`ResponseAPIUsage` (`litellm/types/llms/openai.py`) inherits `BaseLiteLLMOpenAIResponseObject`, which sets `extra="allow"`, so an unknown provider field survives validation instead of being dropped. `ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage` (`litellm/responses/utils.py`) is the single bridge from Responses-shaped usage to the chat-shaped `Usage`, and it splats `response_api_usage.model_extra` onto the `Usage` it returns, minus the keys it already sets explicitly. Every caller that needs chat-shaped usage goes through that one helper, including `cost_calculator.py`, `litellm_logging.py`, the Responses streaming iterator, and the completions bridge, so a field that reaches `ResponseAPIUsage` reaches cost tracking on all of them at once

Built-in tool cost is decided in `StandardBuiltInToolCostTracking.get_cost_for_built_in_tools` (`litellm/litellm_core_utils/llm_cost_calc/tool_call_cost_tracking.py`), which gates on the normalized usage and the response object, then delegates the rate to the provider through the `get_cost_for_web_search_request` dispatch in `litellm/llms/__init__.py`. Per-provider arithmetic lives in `litellm/llms/<provider>/cost_calculator.py`

So, to bill a new provider usage field: read it in that provider's cost calculator, register the provider in the dispatch if it isn't there, and add the gate condition in the shared tracker if the existing gates don't fire. Do not add a provider branch to the shared cost path, do not change a transform to hand cost tracking a different response shape, and do not add a second pass-through for usage on one endpoint

The response the caller sees keeps the schema of the API they called, always. `/v1/responses` returns `input_tokens` and `output_tokens` with provider extras carried alongside, `/v1/chat/completions` returns `prompt_tokens` and `completion_tokens`, and the internal normalization is invisible to both

## Alternatives considered

Reshaping the provider's Responses usage into chat `Usage` inside the provider transform, as [#30817](https://github.com/BerriAI/litellm/pull/30817) first did. Rejected: it breaks the client contract of the endpoint being served, and it has to be repeated per provider and per endpoint

Declaring every provider's usage field on `ResponseAPIUsage` and `Usage` as a typed optional. Rejected as the general mechanism, because the type would grow a field per provider quirk and each addition would still need the bridge updated. Fields we bill across providers do get promoted to real typed fields (`server_tool_use`, `prompt_tokens_details`), and provider-only quirks stay as extras

A dedicated side channel for provider metadata, for example carrying tool counts in `_hidden_params` or in the logging payload rather than on usage. Rejected: cost calculators already receive `Usage` and nothing else about the raw response is guaranteed to reach them, so a side channel means two sources of truth for the same number

Letting the shared tracker special-case providers inline. Rejected: it puts provider pricing in a file that every provider shares, which is what the revert in that PR undid

## Consequences

Adding tool or usage billing for a new provider touches that provider's calculator plus, at most, one gate and one dispatch arm, and it lands on all three endpoints at once

Extras are untyped by construction, so a reader must validate what it gets. `_usage_reports_server_side_web_search_calls` checks `isinstance(details, Mapping)` and that the count is a positive `int` before trusting it, and callers should follow that shape rather than reaching for `getattr` and hoping

Extra field names share a namespace with the bridge's explicit arguments. Gemini image usage carries `prompt_tokens` as an extra on `ResponseAPIUsage`, which collided with the bridge's own keyword argument and raised `TypeError` until the exclusion list in `_transform_response_api_usage_to_chat_usage` grew to cover the keys the bridge sets itself. A new provider that names an extra after a standard chat usage field needs that list checked

Because the same helper runs on both the streaming terminal event and the non-streaming response, tests and proof runs must cover both. A field attached only where the non-streaming path assembles usage will silently bill $0 on `"stream": true`

The decision says nothing about request-side parameters. Provider-specific inputs are a separate mechanism (`get_supported_openai_params`, `map_openai_params`, `extra_body`), and it deserves its own ADR when someone next changes it
