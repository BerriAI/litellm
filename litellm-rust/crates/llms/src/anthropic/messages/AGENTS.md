- https://platform.claude.com/docs/en/api/http/messages/create

- This directory owns Anthropic behavior specific to the Messages API
- Flattening or rewriting web-search results is transformation policy and stays here or in Anthropic helpers shared by its operations
- `web_search_result`, `web_search_tool_result_error`, and encrypted-content fields are Messages protocol data. Shared contracts for them belong in `litellm-types`
- Keep schema definitions separate from decisions about flattening, encrypted results, beta requirements, and model capabilities
