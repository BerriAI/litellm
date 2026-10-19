# rules

- OpenRouter at `https://openrouter.ai/api/v1` with the `HTTP-Referer` and `X-Title` attribution headers
- `cache_control` is kept for caching model families and moved into the last content block
- `usage: {include: true}` is always asked for unless the caller set `usage`
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://openrouter.ai/docs/api-reference/chat-completion
- litellm/llms/openrouter/chat/transformation.py
