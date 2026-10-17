# rules

- NIM's hosted API through the OpenAI SDK URL rule: `{api_base}/chat/completions`
- The supported params depend on the model family; `max_completion_tokens` goes as `max_tokens`
- No `stream_options`, so stream usage is only what NIM sends unasked
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://docs.api.nvidia.com/nim/reference/llm-apis
- litellm/llms/nvidia_nim/chat/transformation.py
