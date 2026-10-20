# rules

- Python pair: `litellm/llms/anthropic/pass_through/adapters/` (`LiteLLMAnthropicMessagesAdapter`, `AnthropicStreamWrapper`)
- Translates a Messages request into the OpenAI Chat Completions wire body a chat provider takes, and that provider's JSON or SSE answer back into Messages. Provider auth, URLs and param rules stay with the chat configs the caller runs the body through
- The chat body is OpenAI's own wire format, not LiteLLM's internal chat shape: no `thinking_blocks`, `cache_control` or Anthropic-only params reach it, since an OpenAI passthrough forwards the body as is
- `max_tokens` is sent as `max_completion_tokens`, which every current OpenAI model takes and the chat configs rename for hosts that only know `max_tokens`
- Chat answers are read through private lenient projections, not `ChatCompletionsResponse` or `ChatCompletionChunk`, because those are LiteLLM's normalized contracts and reject the usage objects real hosts send
