# references

These links describe shared format fields and discriminators. The official API reference is authoritative, and SDK sources only clarify shapes the reference leaves implicit. These contracts are partial typed projections, not exhaustive upstream schemas

## chat_completions.rs and chat_completions/content.rs

- https://developers.openai.com/api/reference/resources/chat

SDK wire definitions:

- https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_content_part_param.py

## responses/output.rs and responses/streaming_websocket.rs

- https://developers.openai.com/api/reference/resources/responses

SDK wire definitions:

- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_output_item.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/mcp_tool_call_error.py

Messages references live in `messages/AGENTS.md`. OCR geometry, table, and key/value contracts describe the existing normalized payload shapes; they do not promise native provider schema parity
