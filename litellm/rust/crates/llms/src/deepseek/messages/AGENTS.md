# rules

- DeepSeek's Anthropic-compatible host: reuses `anthropic/messages` shaping with `x-api-key` auth
- Drops the explicit `type: custom` tool discriminator and Claude Code billing system blocks, which DeepSeek rejects
- Peels OpenAI-style suffixes off a configured base before appending `/anthropic/v1/messages`
- Retries HTTP errors through `retry_anthropic_messages_after_http_error`, not the Anthropic config, so it never gets the Claude thinking-binding retry

# references

- Python counterpart `litellm/llms/deepseek/messages/transformation.py` (`DeepSeekAnthropicMessagesConfig`)
- https://api-docs.deepseek.com/guides/anthropic_api
