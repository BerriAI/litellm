# references

These links describe the provider-specific wire types in this directory. Shared Messages payload references live in `../formats/messages/AGENTS.md`

## anthropic.rs

`AnthropicBeta`, `BetaSet` and `BetaProvider` represent the `anthropic-beta` header values, their
wire spelling and per-host support

- https://platform.claude.com/docs/en/api/beta-headers.md

## minimax.rs

MiniMax's Messages reference documents its image, video, and mid-conversation system content extensions. Its cache reference documents `cache_control` on those blocks

- https://platform.minimax.io/docs/api-reference/text-chat-anthropic
- https://platform.minimax.io/docs/api-reference/text-chat-anthropic.md
- https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache.md
