# structure

- Eden AI's Anthropic Messages adapter for the `edenai` provider. Eden serves the Messages API for its whole catalog, so the payload is forwarded untranslated

# boundaries

- The `cost` extraction and the auth error stay here. Keep `cost` out of the shared Messages response schema

# invariants

- No model gating: every `edenai` model reaches this adapter, unlike Bedrock Mantle, Vertex, Azure AI and GitHub Copilot, which only accept models containing `claude`
- The model name is sent as given once the `edenai/` route prefix is removed. Eden accepts bare names, `anthropic/<model>` and its `@edenai` alias, so do not rewrite or validate it
- A missing API key is an authentication error raised before any request is sent
- A caller-supplied `authorization` or `x-api-key` header wins over the resolved key
- `cache_control` keeps its `ttl` because Eden supports it, so the openai_like reduction to `{"type": ...}` must stay off
- Betas are merged into one `anthropic-beta` header but never filtered against a provider allowlist
- The Anthropic-only thinking rewrites do not run, because Eden is not treated as having Anthropic thinking semantics
- System blocks carrying `x-anthropic-billing-header` metadata are stripped before sending

# gotchas

- The default base already ends in `/v3`, so the final URL is `/v3/v1/messages`. A base ending in `/v1` loses it before `/v1/messages` is appended, and one already ending in `/v1/messages` is used unchanged
- Eden returns a top-level `cost` next to the Anthropic body. When present it replaces the price map estimate as the call's response cost, and a missing or malformed value is ignored
- Streams carry no `cost` yet, so streamed spend falls back to the price map

- A caller `authorization` or `x-api-key` header is enough on its own. Python still demands a resolved key in that case, but the key would never be sent, so Rust does not fail the call for it

# known gaps

- `EdenAIMessagesConfig::reported_response_cost` reads the `cost`, but nothing calls it yet. `BaseMessagesConfig` has no hook for a provider-reported cost and core's Messages route records no response cost, so core needs both before Eden's figure replaces the price map estimate
- Python also falls back to the global `litellm.api_key`; Rust reads only the explicit key and `EDENAI_API_KEY`
- Python maps upstream errors through `EdenAIException`, which has no Rust counterpart here

# references

## python

- `litellm/llms/edenai/messages/transformation.py` (`EdenAIAnthropicMessagesConfig`)
- `litellm/llms/openai_like/messages/transformation.py` (`JSONProviderAnthropicMessagesConfig`)
- `litellm/llms/edenai/common_utils.py`

## docs

- https://www.edenai.co/docs/api-reference/anthropic-messages/create-anthropic-message
- https://www.edenai.co/docs/api-reference/anthropic-messages/create-anthropic-message.md
- https://www.edenai.co/docs/llms.txt
