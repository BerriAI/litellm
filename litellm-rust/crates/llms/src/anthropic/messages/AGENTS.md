# structure

- Anthropic's implementation of the Messages adapter contract in `base_llm/messages`, reached by `custom_llm_provider == "anthropic"` for any model through `ANTHROPIC_MESSAGES_CONFIG` in `inference-messages/src`

# boundaries

- Payload shaping, metadata filtering, tool-ID rewriting, web-search replay handling, thinking translation and beta selection are Anthropic policy and stay here or in Anthropic helpers
- Azure reuses the shaping, request transformation and beta merge, and Bedrock reuses the shaping. That reuse does not make this policy part of the shared Messages contract
- Shared Anthropic-wire request policies belong here and land with the foundation; provider adapters opt into them without widening thinking helpers beyond this module
- `web_search_result`, `web_search_tool_result_error` and encrypted-content schemas belong to `litellm-llms-types`. Only the decisions about flattening, encrypted results, betas and capabilities live here

# invariants

- An OAuth token (`sk-ant-oat...`), forwarded or passed as `api_key`, is the whole credential: it goes out as a bearer and `x-api-key` is dropped
- A forwarded `x-api-key` or `authorization` header in any casing beats the configured key
- Betas end up as one sorted, deduplicated `anthropic-beta` header, computed after the request is final and merged across every casing the caller sent
- Responses and SSE streams are relayed in the Anthropic wire format without rewriting
- Mid-conversation `system` messages and billing-header system blocks reach the first-party API untouched

# gotchas

- `handler.rs` holds pure payload shaping despite its name. New shaping still belongs with transformations

# known gaps

- Workload identity federation (Python `litellm/llms/anthropic/wif.py`) is not a credential source

# references

## python

- `litellm/llms/anthropic/pass_through/messages/transformation.py` (`AnthropicMessagesConfig`)
- `litellm/llms/anthropic/common_utils.py` (auth, URL, OAuth and beta helpers)

## docs

- https://platform.claude.com/docs/en/api/http/messages/create
- https://platform.claude.com/docs/en/api/messages/create.md
- https://platform.claude.com/docs/en/api/beta-headers.md
- https://platform.claude.com/docs/en/api/versioning.md
- https://platform.claude.com/docs/en/build-with-claude/streaming.md
- https://platform.claude.com/docs/en/manage-claude/authentication.md
- https://platform.claude.com/llms.txt
