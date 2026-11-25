# rules

- Eden AI serves the Anthropic Messages API for every model in its catalog: forward the payload untranslated, as Python's `JSONProviderAnthropicMessagesConfig` subclass does
- Reuse Anthropic beta header selection, billing system block stripping and the HTTP error retry
- Reuse `openai_like/messages` for the `/v1/messages` URL and the portable `cache_control` core
- `EDENAI_API_KEY` is required even when the caller forwards an Anthropic credential, which then goes out instead of the bearer key
- Report the per-request `cost` of a complete response. Streams carry no cost, so they fall back to the price map

# references

- Python counterpart `litellm/llms/edenai/messages/transformation.py` (`EdenAIAnthropicMessagesConfig`)
- https://www.edenai.co/docs/api-reference/anthropic-messages/create-anthropic-message
