# AGENTS.md

litellm-rust has exactly THREE crates. A crate is a LAYER, not a route. Routes (ocr, realtime, chat) and providers (mistral, openai) are MODULES inside the layers.

## Crates

| Crate | Role | Pure / I/O |
|-------|------|------------|
| litellm-core | Translation layer — types, route contracts (traits), provider transforms (modules under providers/), and the router. Builds requests/responses; no network. | Pure |
| litellm-ai-gateway | Routes + host — the only crate that touches the network. HTTP/WebSocket I/O (modules under io/) plus the axum server binary (behind the `server` feature). | I/O |
| litellm-python-bridge | PyO3 cdylib exposing Rust to the litellm Python SDK — a thin adapter over litellm-ai-gateway's I/O. | Binding |

Dependency direction (acyclic): litellm-core ← litellm-ai-gateway ← litellm-python-bridge.

Adding a crate: default to a MODULE. New crate ONLY on a real trigger — separate artifact (binary/cdylib), proc-macro, shared foundation, or publishable standalone. A new provider or route is none of these.

Adding a crate fails crates/core/tests/workspace_crate_allowlist.rs until you update its allowlist and this file — intentional.

## Style

All Rust in `litellm-rust/` follows the official Rust Style Guide:
https://doc.rust-lang.org/style-guide/

`rustfmt` implements its formatting by default, so run `cargo fmt` before committing; CI gates every PR on `cargo fmt --check`. Do not hand-format against rustfmt or add a `rustfmt.toml` that diverges from the default style.

Beyond formatting, follow the guide's naming and idiom conventions rustfmt cannot auto-apply: `snake_case` items/functions/modules, `UpperCamelCase` types/traits/variants, `SCREAMING_SNAKE_CASE` constants/statics (acronyms as one word, e.g. `HttpClient`), and the import grouping and item ordering it prescribes. See CLAUDE.md for the detailed version.
