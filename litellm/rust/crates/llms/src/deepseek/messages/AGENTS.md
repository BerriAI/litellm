# rules

- DeepSeek's Anthropic-compatible host: reuses `anthropic/messages` shaping with `x-api-key` auth
- Drops the explicit `type: custom` tool discriminator and Claude Code billing system blocks, which DeepSeek rejects
- Peels OpenAI-style suffixes off a configured base before appending `/anthropic/v1/messages`

# references

- https://api-docs.deepseek.com/guides/anthropic_api
