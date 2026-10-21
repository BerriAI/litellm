# rules

- Owns Anthropic provider policy: credentials, OAuth handling, endpoint resolution, beta requirements, model capabilities and transformations
- Paths, header names and default headers come from `litellm_llms_types::providers::anthropic`
- Choose auth and required headers here. Shared infrastructure applies them
- `ReplayedWebSearchResult` and `ReplayedWebSearchContent` are private partial models for replay flattening. Keep them private while they only serve that transformation

# references

- https://platform.claude.com/docs/en/api/overview.md
- https://platform.claude.com/docs/en/api/errors.md
