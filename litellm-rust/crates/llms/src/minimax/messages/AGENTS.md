# structure

- MiniMax's Messages adapter for its Anthropic-compatible `/v1/messages` API, reached by every `minimax/<model>` Messages call with no model-name gating, unlike the Claude-only Bedrock Mantle, Vertex AI and Azure AI branches

# invariants

- Never send an Anthropic credential to MiniMax. Python silently falls back to `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` when no MiniMax key exists; do not port that, and name MiniMax in the missing-key error
- The caller's `thinking` and `output_config` reach MiniMax unchanged, because MiniMax does not use Anthropic thinking semantics. Only `reasoning_effort` is still mapped to native params first
- `x-anthropic-billing-header` text blocks are stripped from `system`, and `system` is omitted when nothing remains
- No `anthropic-beta` value is forwarded, because the beta allowlist has no `minimax` entry

# gotchas

- A China-region base such as `https://api.minimaxi.com/anthropic`, a base ending in `/v1`, and one already ending in `/v1/messages` must all resolve to the same `/v1/messages` path
- Environment validation receives the caller's `api_base`, not the resolved default
- `compaction` is unsupported because the provider is not `anthropic`, even though the rest of the payload policy is Anthropic's. Python still sends it in the body; Rust drops it silently, like Tencent, and the compaction beta never goes out

- Reuse the Anthropic request shaping and response/SSE decoding from `anthropic/messages` explicitly, and keep only MiniMax's differences here, since Python's config is a thin subclass of Anthropic's
- MiniMax-only content blocks (image, video, mid-conversation system) live in `litellm-llms-types::providers::minimax`, not here

# known gaps

- Python's last key fallback to the global `litellm.api_key` has no Rust equivalent
- Not wired into `inference-messages` (`inference-messages/src/common_utils.rs`) or `LlmProviders` yet
- The MiniMax-only content blocks in `litellm-llms-types::providers::minimax` are not parsed or validated by the adapter; they pass through as unrecognized Anthropic blocks

# references

## python

- `litellm/llms/minimax/messages/transformation.py` (`MinimaxMessagesConfig`)
- `litellm/llms/anthropic/pass_through/messages/transformation.py` (`AnthropicMessagesConfig`, where most behavior is inherited)
- `ProviderConfigManager._get_provider_anthropic_messages_config_cached` in `litellm/utils.py`

## docs

- https://platform.minimax.io/docs/api-reference/text-chat-anthropic
- https://platform.minimax.io/docs/api-reference/text-chat-anthropic.md
- https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache.md
- https://platform.minimax.io/docs/llms.txt
