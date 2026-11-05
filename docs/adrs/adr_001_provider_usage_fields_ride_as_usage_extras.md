# ADR 001: Provider usage fields ride as extras on the normalized usage object

Status: Accepted

## Context

Providers report usage fields that no other provider has, such as xAI's `usage.server_side_tool_usage_details.web_search_calls` or Anthropic's `usage.inference_geo`, and cost tracking has to read them or that spend never reaches `LiteLLM_SpendLogs`. One provider response can be served on `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`, streaming or not, so a field that only one of those paths carries is billed wrong on the others

[#30817](https://github.com/BerriAI/litellm/pull/30817) fixed unbilled xAI web searches. One draft did the copy in the xAI Responses transform to keep provider code out of shared files, and that swapped `/v1/responses` usage to the chat shape, so callers of that endpoint lost the Responses schema. The merged version keeps the schema, copies unknown usage fields in the shared Responses-to-chat bridge for every provider, and gates on the reported field without naming the provider

In October 2026, `Usage` and `ResponseAPIUsage` kept unknown usage keys as pydantic extras, and that bridge copied them. Paths that build usage from named fields kept only the fields they name: the Anthropic, Gemini, and Bedrock Converse transforms, the stream builder, and the bridges from chat usage to `/v1/responses` and `/v1/messages`

## Decision

Provider-only usage fields stay on the usage object as pydantic extras, and cost tracking reads them there, in that provider's cost calculator or in a shared gate that checks the field instead of the provider. An extra reaches callers as an added usage key. No transform renames, removes, or reshapes a usage field the endpoint's schema defines to feed cost tracking, though it may fill one LiteLLM's usage object already defines, such as `server_tool_use`

A shared path that rebuilds usage, such as the stream builder or a bridge between endpoint shapes, copies the extras instead of naming provider fields. A provider transform that builds usage from named fields names the new field. Neither is bypassed with a side channel. Older code that differs, such as the Azure branches in the shared tracker and the Gemini pricing tier read from `_hidden_params`, predates this ADR

## Alternatives Considered

**Reshape usage in the provider transform**, as a draft of #30817 did. It changes the schema that callers of that endpoint rely on, and every provider and endpoint needs its own copy

**A typed field on `ResponseAPIUsage` and `Usage` for each provider's quirk**. The types grow without bound and every bridge needs an update for each one. A field several providers report, like `server_tool_use`, does get one

**A side channel such as `_hidden_params`**. Cost calculators receive `Usage`, so a side channel makes a second source of truth for the same number

**Provider branches or provider field names in shared code**. They put one provider's logic in a file every provider shares, which is why #30817 replaced its xAI-only gate with one that checks the reported field

## Consequences

Benefits:

- Billing a new provider usage field touches that provider's calculator (plus a `cost_per_token` arm when the calculator is new), its transform when that names fields, and for a built-in tool at most one gate and one `get_cost_for_web_search_request` arm

Costs:

- Extras are untyped, so a reader validates them before use, as `_usage_reports_server_side_web_search_calls` does
- The stream builder and the bridges from chat usage did not copy extras when this was written, so until they do, a field billed on those paths is named in them, and tests and proof runs cover each endpoint and `"stream": true`
- This covers usage fields LiteLLM prices, not request-side params or a provider's own charge for the request, which OpenRouter and opencode Go report and `response_cost_calculator` reads from `_hidden_params` first
