# rules

- Holds only the free functions JSON-configured Messages hosts reuse: the `/v1/messages` URL from a base and the reduction of `cache_control` to its portable `{"type": ...}` core
- Python's `OpenAILikeAnthropicMessagesConfig` is not ported, so `openai_like` is not a Messages provider. Porting it adds its own `impl BaseMessagesConfig` here, built on these helpers

# references

- Python counterpart `litellm/llms/openai_like/messages/transformation.py` (`OpenAILikeAnthropicMessagesConfig`, `JSONProviderAnthropicMessagesConfig`)
