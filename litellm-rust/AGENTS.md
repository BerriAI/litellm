# AGENTS.md

litellm-rust has exactly FOUR crates. A crate is a LAYER, not a route. Routes and providers are modules inside the layers. `litellm-runtime` is an intentional reusable execution layer, introduced OCR-first.

## Crates

| Crate | Role |
|-------|------|
| litellm-core | Provider transformations and shared types. OCR core code is deterministic transformation only. Existing non-OCR route entrypoints remain in core. |
| litellm-runtime | Reusable OCR execution: provider/model and config selection, environment/auth, URL/headers, document materialization, HTTP, polling, and lifecycle ordering. |
| litellm-ai-gateway | The axum server and WebSocket hosts. Adapts host logger, guardrail, and metadata types to runtime interfaces. |
| litellm-python-bridge | PyO3 cdylib exposing Rust to the Python SDK. OCR calls runtime directly; unrelated legacy paths may still call gateway or core. |

Dependency direction (acyclic): ai-gateway/python-bridge -> runtime -> core. Python bridge retains an ai-gateway dependency for unrelated audio and WebSocket code.

## Where a route lives

A top-level LiteLLM call is a module under `crates/core/src/<route>/`, shaped like `messages`:

```
core/src/messages/
  mod.rs             # pub async fn messages(..) -> CoreResult<..>  (+ messages_stream for SSE)
  types.rs           # request/response types, MessagesRequest
  transformation.rs  # the provider template trait
  prepare.rs         # provider resolution, auth headers, URL
  handler.rs         # the provider call
  client.rs          # the shared reqwest client
```

OCR provider handlers live in `runtime`, not `ai-gateway`. This boundary is OCR-first; audio, realtime, Messages, Chat, Responses, and generic lifecycle code do not move with it.

Adding a crate: default to a MODULE. New crate ONLY on a real trigger — separate artifact (binary/cdylib), proc-macro, shared foundation, or publishable standalone. A new provider or route is none of these.

Adding a crate fails crates/core/tests/workspace_crate_allowlist.rs until you update its allowlist and this file — intentional.

## Style

All Rust in `litellm-rust/` follows the official Rust Style Guide:
https://doc.rust-lang.org/style-guide/

`rustfmt` implements its formatting by default, so run `cargo fmt` before committing; CI gates every PR on `cargo fmt --check`. Do not hand-format against rustfmt or add a `rustfmt.toml` that diverges from the default style.

Beyond formatting, follow the guide's naming and idiom conventions rustfmt cannot auto-apply: `snake_case` items/functions/modules, `UpperCamelCase` types/traits/variants, `SCREAMING_SNAKE_CASE` constants/statics (acronyms as one word, e.g. `HttpClient`), and the import grouping and item ordering it prescribes. See CLAUDE.md for the detailed version.
