# rules

- Send the native Messages payload to `/api/v1/messages` and relay Anthropic SSE
- Authenticate with a bearer token, never treat `x-api-key` as an OpenRouter credential
- Keep cache TTLs, routing fields, metadata, and provider usage extensions
- Strip first-party billing system blocks using the Anthropic helper

# references

- https://openrouter.ai/docs/api/api-reference/anthropic-messages/create-messages
