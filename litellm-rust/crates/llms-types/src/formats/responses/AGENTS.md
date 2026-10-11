# rules

- Responses originated at OpenAI and is now offered by other hosts and gateways. Types here describe the format, not OpenAI's deployment of it
- Served over HTTP and over the WebSocket mode. `ResponsesWsEvent` is a wire event and lives here
- `ResponsesWsTransformResult` wraps a provider transformation's output, so it lives in `llms/src/base_llm/responses/`
- Hosts that accept only part of the format, or add output item types, are handled in `llms/src/<provider>/responses/`. Keep unknown output items passing through instead of rejecting them
- Responses-to-Chat emulation and model-specific parameter rewriting are transformations, not format rules

# references

- https://developers.openai.com/api/reference/resources/responses
- https://developers.openai.com/api/reference/resources/responses/streaming-events
- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_output_item.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/mcp_tool_call_error.py
