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
