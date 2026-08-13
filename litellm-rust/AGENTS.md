# AGENTS.md

litellm-rust has exactly THREE crates. A crate is a LAYER, not a route. Routes (ocr, realtime, chat) and providers (mistral, openai) are MODULES inside the layers.

## Crates

| Crate | Role |
|-------|------|
| litellm-core | The LiteLLM SDK in Rust. One public entrypoint per top-level call (`messages::messages()`), owning types, transforms, provider resolution, auth, and the provider HTTP call. Call it, get a typed response. |
| litellm-ai-gateway | The axum server (behind the `server` feature) plus the WebSocket hosts. Translates HTTP/WS to core entrypoints; owns no provider logic and no handlers. |
| litellm-python-bridge | PyO3 cdylib exposing Rust to the litellm Python SDK — marshals Python objects and calls core entrypoints. |

Dependency direction (acyclic): litellm-core ← litellm-ai-gateway ← litellm-python-bridge.

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

Handlers never live in `ai-gateway`. `ocr`, `audio_transcription`, and `realtime` are still hosted there from before this rule; they move to `core` as they are touched.

Adding a crate: default to a MODULE. New crate ONLY on a real trigger — separate artifact (binary/cdylib), proc-macro, shared foundation, or publishable standalone. A new provider or route is none of these.

Adding a crate fails crates/core/tests/workspace_crate_allowlist.rs until you update its allowlist and this file — intentional.

## Style

All Rust in `litellm-rust/` follows the official Rust Style Guide:
https://doc.rust-lang.org/style-guide/

`rustfmt` implements its formatting by default, so run `cargo fmt` before committing; CI gates every PR on `cargo fmt --check`. Do not hand-format against rustfmt or add a `rustfmt.toml` that diverges from the default style.

Beyond formatting, follow the guide's naming and idiom conventions rustfmt cannot auto-apply: `snake_case` items/functions/modules, `UpperCamelCase` types/traits/variants, `SCREAMING_SNAKE_CASE` constants/statics (acronyms as one word, e.g. `HttpClient`), and the import grouping and item ordering it prescribes. See CLAUDE.md for the detailed version.
