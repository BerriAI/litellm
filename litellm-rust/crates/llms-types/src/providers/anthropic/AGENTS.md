# rules

- Shared across Anthropic's Messages, chat, count-tokens and batches adapters: API base, endpoint paths, header names, API version and default headers
- `beta.rs` holds `AnthropicBeta`, `BetaSet` and `BetaProvider`, the `anthropic-beta` values, their wire spelling and which hosts accept each one
- Choosing which betas a request needs is policy in `llms/src/anthropic/`

# references

- https://platform.claude.com/docs/en/api/beta-headers.md
