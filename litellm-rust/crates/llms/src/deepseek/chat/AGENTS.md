# rules

- DeepSeek at `https://api.deepseek.com/beta`; `/chat/completions` is appended unless the base ends with it
- `thinking` and `reasoning_effort` become the `thinking` toggle, only function tools are sent
- A thinking turn replays `reasoning_content` on every assistant message
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://api-docs.deepseek.com/api/create-chat-completion
- litellm/llms/deepseek/chat/transformation.py
