# CLAUDE.md

Rules for `litellm-rust/crates/core`.

## Responsibility

`core` is the LiteLLM SDK in Rust: it makes the LLM call. Every top-level
LiteLLM call has a public entrypoint here, named after the route
(`messages::messages()` is the Rust equivalent of `litellm.messages()`), and
calling it returns a typed non-streaming response.

Allowed:
- The public entrypoint for a route, plus its `<route>_stream` variant when the
  route supports streaming.
- Provider resolution, auth header construction, URL building, and the provider
  HTTP call (shared reused client, connect + request timeouts).
- Shared request/response structs.
- Typed errors with stable, non-sensitive messages.
- Deterministic validation helpers.
- Serialization helpers that intentionally mirror Python output shape.
- Route templates that match Python base config responsibilities, such as
  `messages::transformation::AnthropicMessagesProviderConfig`.

Not allowed:
- Serving HTTP: axum routers, extractors, and other transport concerns.
- Filesystem, database, or cache access.
- Config file reading or rollout state; the host resolves those and passes them
  in. Env reads are limited to credential fallback in a route's `prepare.rs`.
- Logging callbacks, tracing spans, spend writes, or customer callbacks.
- Provider-specific branching that belongs in `providers`.
- Panics for user/provider-controlled input.

## Typed Contracts (core rule)

Trait and function boundaries MUST be strongly typed. No stringly-typed JSON
(`&str` / `String` / `Vec<String>` / bare `serde_json::Value`) as a transform
input or output. Parse wire bytes into typed structs/enums at the host edge;
`core` and `providers` operate only on those types (e.g. `RealtimeEvent`,
`RealtimeTransformResult`, `OcrRequestData`). A `type`-style discriminator is a
typed field on a struct, not a raw string threaded through the API.

## Structure

Use route names directly under `src/`: `messages`, `ocr`, future
`chat_completions`, `embeddings`, and similar top-level LiteLLM calls. Do not
invent broad names like `engine` for route contracts.

`src/messages` is the reference shape for a route module:

```
mod.rs             pub async fn messages(..) (+ messages_stream)
types.rs           request/response types
transformation.rs  the provider template trait
prepare.rs         provider resolution, auth headers, URL
handler.rs         the provider call
client.rs          the shared reqwest client
```

## Parity Rules

- Every shared type used by a provider transform needs unit tests for
  serialization shape.
- If Python parity requires always emitting a `null` field instead of omitting
  it, document that in code and pin it with a test.
- Error enums should preserve enough detail for Python/HTTP hosts to map errors
  consistently without exposing document contents or upstream bodies.
