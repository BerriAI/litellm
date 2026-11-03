# rules

- Chat Completions originated at OpenAI and is the most widely cloned LLM API. Types here describe the format, not OpenAI's deployment of it
- Spoken natively by OpenAI and OpenAI-compatible hosts, served by `llms/src/openai_like/chat/`
- Other providers translate into it from their own API (Anthropic Messages, Bedrock Converse) in `llms/src/<provider>/chat/`
- Compatible hosts commonly
  - accept a subset of parameters, which is recorded per provider in `supported_openai_param_mappings`, not by trimming the type
  - add top-level request or response fields, preserved through the existing extra-field maps instead of new typed fields
  - return slightly different usage or finish reasons, normalized in the provider transformation
- `ChatCompletionsResponse` is LiteLLM's response contract to the host, a typed projection rather than the full upstream schema

# references

- https://developers.openai.com/api/reference/resources/chat
- https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_content_part_param.py
