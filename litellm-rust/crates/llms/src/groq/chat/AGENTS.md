# rules

- Groq at `https://api.groq.com/openai/v1`
- A JSON schema on a model without native structured outputs goes as a forced `json_tool_call` tool, refused beside caller tools
- A streamed `response_format` is refused: Python fakes that stream from a complete call
- `web_search_options` adds the `browser_search` tool; `service_tier` in the answer is held to OpenAI's values
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://console.groq.com/docs/api-reference#chat
- litellm/llms/groq/chat/transformation.py
