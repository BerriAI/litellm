# rules

- xAI at `https://api.x.ai/v1`
- `web_search_options` and non-user `name`s are dropped, tools lose `strict`
- Reasoning tokens xAI counts apart are folded into `completion_tokens`, in answers and stream chunks
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://docs.x.ai/docs/api-reference#chat-completions
- litellm/llms/xai/chat/transformation.py
