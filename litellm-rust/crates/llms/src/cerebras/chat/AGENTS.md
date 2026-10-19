# rules

- Cerebras through the OpenAI SDK URL rule at `https://api.cerebras.ai/v1`
- `reasoning_effort` only for models the cost map marks as reasoning; `max_completion_tokens` goes as `max_tokens`
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://inference-docs.cerebras.ai/api-reference/chat-completions
- litellm/llms/cerebras/chat.py
