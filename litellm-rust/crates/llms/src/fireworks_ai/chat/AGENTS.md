# rules

- Fireworks AI at `https://api.fireworks.ai/inference/v1`, the model named by its full resource (`resolve_fireworks_resource_name`)
- Tool, tool choice and reasoning params are gated by the cost map, file parts and images on non-vision models are rejected
- vLLM/NIM `extra_body` keys are translated (`truncate_prompt_tokens`, `chat_template_kwargs`, `guided_*`) or dropped
- Tool calls Fireworks writes into the content go back as `tool_calls`; the answer's model gets the `fireworks_ai/` prefix
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://docs.fireworks.ai/api-reference/post-chatcompletions
- litellm/llms/fireworks_ai/chat/transformation.py
