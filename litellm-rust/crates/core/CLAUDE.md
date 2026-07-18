# CLAUDE.md

Rules for `litellm-rust/crates/core`.

## Responsibility

`core` owns shared data types, typed errors, and deterministic helper contracts.
It must stay pure and host-independent.

Allowed:
- Shared request/response structs.
- Typed errors with stable, non-sensitive messages.
- Deterministic validation helpers.
- Serialization helpers that intentionally mirror Python output shape.
- Route templates that match Python base config responsibilities, such as
  `ocr::transformation::OcrProviderConfig`.

Not allowed:
- Network, filesystem, database, cache, or environment access.
- Secret reads or auth/header construction.
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

Use route names directly under `src/`: `ocr`, future `messages`,
`chat_completions`, `embeddings`, and similar top-level LiteLLM calls. Do not
invent broad names like `engine` for route contracts.

## Parity Rules

- Every shared type used by a provider transform needs unit tests for
  serialization shape.
- If Python parity requires always emitting a `null` field instead of omitting
  it, document that in code and pin it with a test.
- Error enums should preserve enough detail for Python/HTTP hosts to map errors
  consistently without exposing document contents or upstream bodies.
