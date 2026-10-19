# rules

- Format spec lives in `llms-types/src/formats/chat_completions/AGENTS.md`
- Passthrough, not a `BaseConfig`: that contract parses messages and normalizes the response into `ChatCompletionsResponse`, which drops tool calls, ids and every field it does not type. OpenAI's own body and SSE bytes reach the caller as sent
- The request changes only by the deployment model name, the `extra_body` merge the OpenAI SDK does, and dropping the LiteLLM-owned keys Python consumes before the call
- Endpoint and credential fallbacks are shared with Responses in `../common_utils.rs`
- `compatible.rs` is the request flow and response fixups the OpenAI-compatible providers share. Each provider's config lives in `src/<provider>/chat/`, and its body and SSE bytes change only where its config's fixups change them
