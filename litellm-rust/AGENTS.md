# litellm-rust AGENTS.md

litellm-rust has exactly THREE crates. A crate is a LAYER, not a route. Routes (ocr, realtime, chat) are MODULES inside the layers.

## The three crates

| Crate | Role | Pure/IO | May depend on |
|-------|------|---------|---------------|
| litellm-core | Translation layer: types + route contracts (traits) + provider transforms | Pure | (nothing external beyond serde/thiserror) |
| litellm-ai-gateway | Routes/host: all network I/O + end-to-end route functions | I/O | litellm-core |
| litellm-python-bridge | PyO3 cdylib exposing Rust to the litellm Python SDK | Binding | litellm-ai-gateway, litellm-core |

## Adding a crate

Default to a MODULE. Add a new crate ONLY when one trigger fires: (a) it must compile to a separate artifact (binary / cdylib), (b) it is a proc-macro (forced), (c) it is a foundation layer many crates share, or (d) it must be publishable/usable standalone. None of these is true for a new provider or route — those are modules.

## Enforcement

Adding a crate WILL fail the enforcement test at `crates/core/tests/workspace_crate_allowlist.rs` until you update its allowlist and this file — that gate is intentional.
