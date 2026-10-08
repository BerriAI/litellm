# references

These links describe shared format fields and discriminators. The new contracts are partial typed projections of the reference branch's payloads, not exhaustive upstream schemas. Optional fields accept missing values and nulls, and omit both on serialization. Known malformed fields and unknown tagged variants are rejected; extension maps retain arbitrary data. Existing request and response types retain their parsing behavior

## chat_completions.rs and chat_completions/content.rs

- https://developers.openai.com/api/reference/resources/chat
- https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_content_part_param.py

## responses/output.rs and responses/streaming_websocket.rs

- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_output_item.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/mcp_tool_call_error.py

Messages references live in `messages/AGENTS.md`. OCR geometry, table, and key/value contracts describe the existing normalized payload shapes; they do not promise native provider schema parity
