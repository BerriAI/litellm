# AGENTS.md

litellm-rust has six crates. A crate is a layer, shared foundation, or separately built host, not a route. Routes (ocr, realtime, chat) and providers (mistral, openai) are modules inside the layers.

## Crates

| Crate | Role |
|-------|------|
| litellm-core | The LiteLLM SDK in Rust. One public entrypoint per top-level call (`messages::messages()`), owning types, transforms, provider resolution, auth, and the provider HTTP call. Call it, get a typed response. |
| litellm-config | Config-loading boundary. Returns resolved core deployment data and optionally delegates loading to Python. |
| litellm-gateway-inference | Framework-independent gateway runtime and integrations shared by the server and Python bridge. Owns transport-neutral orchestration and legacy call modules, but no Axum or Tower dependencies. |
| litellm-gateway-server | The root Axum binary and composition crate. Owns route composition, auth extractors, application state, startup config, and HTTP-only Tower dependencies, then delegates domain behavior to extracted gateway crates. |
| litellm-python-interop | Domain-neutral PyO3 foundation for GIL handling and typed Python/Serde conversion. |
| litellm-python-bridge | PyO3 cdylib exposing LiteLLM Rust APIs to the Python SDK. Owns API registration, domain wiring, and Python exception mapping. |

Dependency direction is acyclic: `litellm-config` depends on `litellm-core`; `litellm-gateway-inference` depends on core; `litellm-gateway-server` depends on inference, core, and config; and `litellm-python-bridge` depends on the reusable domain layers and `litellm-python-interop`. Reusable crates must not depend on `litellm-gateway-server`. The interop foundation depends on no LiteLLM domain crate.

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

Provider handlers never live in `gateway-server`. `ocr`, `audio_transcription`, and realtime provider I/O are still hosted in `gateway-inference` from before this rule; they move to `core` as they are touched.

Adding a crate: default to a module. A new crate requires a real trigger: separate artifact (binary/cdylib), proc-macro, shared foundation, or publishable standalone. A new provider or route is none of these.

Adding a crate fails crates/core/tests/workspace_crate_allowlist.rs until you update its allowlist and this file — intentional.

## Style

All Rust in `litellm-rust/` follows the official Rust Style Guide:
https://doc.rust-lang.org/style-guide/

`rustfmt` implements its formatting by default, so run `cargo fmt` before committing; CI gates every PR on `cargo fmt --check`. Do not hand-format against rustfmt or add a `rustfmt.toml` that diverges from the default style.

Beyond formatting, follow the guide's naming and idiom conventions rustfmt cannot auto-apply: `snake_case` items/functions/modules, `UpperCamelCase` types/traits/variants, `SCREAMING_SNAKE_CASE` constants/statics (acronyms as one word, e.g. `HttpClient`), and the import grouping and item ordering it prescribes. See CLAUDE.md for the detailed version.
