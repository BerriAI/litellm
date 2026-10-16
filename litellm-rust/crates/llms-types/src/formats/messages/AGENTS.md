# rules

- Messages originated at Anthropic and is served by many hosts. Types here describe the format, independent of any host
- Spoken by Anthropic, Bedrock InvokeModel, Vertex AI `rawPredict`, Azure AI Foundry, DeepSeek and MiniMax, each adapted in `llms/src/<provider>/messages/`
- Hosts extend it with
  - their own content blocks or media sources, which get typed in `providers/<provider>/` once a consumer needs them (MiniMax image, video and mid-conversation system blocks)
  - beta features gated by the `anthropic-beta` header, whose accepted values differ per host (`providers/anthropic/beta.rs`)
- Hosts deviate by
  - moving `model` or the API version out of the body or into the URL (Bedrock, Vertex)
  - framing the stream differently (AWS event stream on Bedrock instead of SSE)
  - rejecting fields or discriminators (DeepSeek rejects `type: custom` tools and billing system blocks)
  - Handle these in the provider adapter, never by loosening or forking a type here
- Request and response bodies, messages, content blocks, tools, usage and stream-event payloads live here
  - Web-search results, encrypted-content fields and thinking configuration are data here
  - Flattening results, stripping encrypted content, choosing thinking budgets and requiring betas are provider policy in `llms`
- `MessagesTransformContext` and other adapter inputs live in `llms/src/base_llm/messages/`. Call envelopes live in `inference-messages`
- Beta-only blocks and tools (MCP, compaction, tool changes, advisor, fallback, browser state, toolsets) follow the beta reference

# references

- https://platform.claude.com/docs/en/api/http/messages/create
- https://platform.claude.com/docs/en/api/messages/create.md
- https://platform.claude.com/docs/en/api/beta/messages/create.md
- https://platform.claude.com/docs/en/build-with-claude/streaming.md
- https://platform.claude.com/docs/en/build-with-claude/context-editing
