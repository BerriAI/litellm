# ADR 001: Provider usage fields ride as extras on the normalized usage object

Status: Accepted

## Context

Providers report usage fields that no other provider has, such as xAI's `usage.server_side_tool_usage_details.web_search_calls` or Anthropic's `usage.inference_geo`, and cost tracking has to read them or that spend never reaches `LiteLLM_SpendLogs`. One provider response can be served on `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`, streaming or not, so a field that only one of those paths carries is billed wrong on the others

[#30817](https://github.com/BerriAI/litellm/pull/30817) fixed unbilled xAI web searches. It first copied xAI's field in the shared bridge from Responses usage to chat `Usage`, then made that copy generic, then moved it into the xAI Responses transform to keep provider code out of shared files. That transform swapped `/v1/responses` usage to the chat shape, so callers of that endpoint lost the Responses schema. It also added an xAI-only gate to the shared built-in tool tracker and then removed it. The merged version keeps the Responses schema, copies unknown usage fields in the shared bridge for every provider, and gates on the reported field without naming the provider

In October 2026, when this was written, `Usage` (`litellm/types/utils.py`) kept every keyword argument it does not define. A chat transform that hands the provider response to `convert_to_model_response_object`, as the OpenAI-compatible ones do, kept the provider's unknown usage keys, and one that builds `Usage` from named fields, as the Anthropic, Gemini, and Bedrock Converse transforms do, kept only the fields it names. `ResponseAPIUsage` allowed extra fields, and `ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage` (`litellm/responses/utils.py`), the one bridge from Responses usage to chat `Usage`, copied them across minus a fixed exclusion list. Streamed chat (`ChunkProcessor.calculate_usage`) rebuilt usage from named fields, which included Anthropic's `inference_geo` and `speed`, and lost every other extra. `/v1/responses` served by a chat-only provider (`transform_chat_completion_usage_to_responses_usage`) kept only token counts and a few named fields, so it lost the extras and `server_tool_use`

## Decision

Provider-only usage fields stay on the usage object as pydantic extras, and cost tracking reads them there, in that provider's cost calculator or in a shared gate that checks the field instead of the provider. No transform renames, removes, or reshapes the usage fields an endpoint's schema defines in order to feed cost tracking. A transform may fill a field LiteLLM's usage object already defines, such as `prompt_tokens_details.web_search_requests` or `server_tool_use`

A transform that builds usage from named fields passes a new billed field through, and a path that drops extras is extended to carry them, never bypassed with a side channel

## Alternatives Considered

**Reshape usage in the provider transform**, as #30817 did before it merged. It changes the response schema that callers of that endpoint rely on, and every provider and endpoint needs its own copy

**A typed field on `ResponseAPIUsage` and `Usage` for each provider's quirk**. The types grow without bound and the bridge still needs an update for each one. A field that several providers report, like `server_tool_use`, does become a typed field

**A side channel such as `_hidden_params`**. Cost calculators receive `Usage`, so a side channel makes a second source of truth for the same number

**Provider branches inside the shared tracker**. They put one provider's logic in a file every provider shares, so #30817 replaced its xAI-only gate with one that checks the reported field. Older Azure branches for file search, vector store, and computer use pricing are still there

## Consequences

Benefits:

- Billing a new provider usage field touches that provider's calculator, plus its transform when that builds usage from named fields, and at most one gate and one dispatch arm

Costs:

- Extras are untyped, so a reader validates them before use, as `_usage_reports_server_side_web_search_calls` does
- The bridge drops an extra whose name is on its exclusion list, so a new extra with one of those names needs that list checked
- A path that rebuilds usage from named fields loses extras until it is extended, so tests and proof runs cover each endpoint and `"stream": true`
- This covers response usage only. Request-side provider params (`get_supported_openai_params`, `map_openai_params`) are a separate mechanism
