# rules

- Together AI at `https://api.together.ai/v1`
- Tool and `response_format` params are refused or dropped only when the `together_ai/<model>` cost map entry rules them out
- `reasoning_effort` is translated per model family, `none` turns reasoning off with `reasoning.enabled: false`
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://docs.together.ai/docs/chat-overview
- litellm/llms/together_ai/chat/transformation.py
