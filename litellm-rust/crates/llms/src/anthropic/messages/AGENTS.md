This directory owns Anthropic's implementation of the Messages adapter contract in `base_llm/messages`. Shared Messages API data contracts belong in `litellm-llms-types::formats::messages`, and call orchestration belongs in `inference-messages`. Sharing the `llms` crate with `base_llm/messages` does not erase this boundary

Payload shaping, metadata filtering, tool-ID rewriting, web-search replay handling, thinking translation, and beta selection are provider policy. Keep them here or in Anthropic helpers shared by its operations. Pure payload shaping belongs with transformations, even if an existing file is named `handler.rs`

Bedrock and Azure adapters may explicitly reuse these helpers where Anthropic policy applies to their Claude backend. That reuse does not make the policy part of the shared Messages contract or a default for every provider. Shared `base_llm` code must never depend on this implementation

`web_search_result`, `web_search_tool_result_error`, and encrypted-content fields are protocol data owned by `litellm-llms-types`. Keep those schemas separate from decisions about flattening, encrypted results, beta requirements, and model capabilities

Protocol reference: [Messages API](https://platform.claude.com/docs/en/api/http/messages/create)
