# rules

- vLLM's OpenAI-compatible server at the deployment's `api_base`; there is no default host, and an unset key sends `fake-api-key`
- Tools lose `additionalProperties: false` and `strict`, custom tools go as function tools, `thinking` becomes `reasoning_effort`
- Replayed assistant turns lose `thinking_blocks` and non-string `reasoning_content`, content lists flatten to text
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- litellm/llms/hosted_vllm/chat/transformation.py
