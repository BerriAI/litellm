This directory owns shared Messages API data contracts and serialization: request and response bodies, messages, content blocks, usage, and stream-event payloads. Messages is a format independent of the provider that originated it. Keep one canonical definition of each shared contract here

Adapter contracts and execution inputs such as `MessagesTransformContext` belong in `llms/src/base_llm/messages`. Provider rewriting and interpretation belong in `llms/src/<provider>/messages`. Call envelopes, live streams, and call orchestration belong in `inference-messages`

Represent web-search results, encrypted-content fields, and thinking configuration as data here. Decisions to flatten results, remove encrypted content, select thinking budgets, or require beta headers belong to provider transformations. Data-shape validation belongs here, while model capability checks and request adaptation do not

# references

These upstream contracts document fields and discriminators. They are documentation links, not saved golden snapshots. Beta-only blocks and tools (MCP, compaction, tool changes, advisor, fallback, browser state, and toolsets) follow the beta reference

## Messages bodies, content, tools, and streaming

- https://platform.claude.com/docs/en/api/http/messages/create
- https://platform.claude.com/docs/en/api/messages/create.md
- https://platform.claude.com/docs/en/api/beta/messages/create.md
- https://platform.claude.com/docs/en/build-with-claude/streaming.md
- https://platform.claude.com/docs/en/build-with-claude/context-editing

Anthropic-compatible hosts document their deviations in the `llms/src/<provider>/messages` guides. Copy a host reference into `../../providers/AGENTS.md` only when that host gets a typed extension in `providers`
