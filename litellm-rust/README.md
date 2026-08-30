# LiteLLM Rust

This workspace contains the staged Rust implementation for LiteLLM.

`litellm-core` is the LiteLLM SDK in Rust: one entrypoint per top-level call
that makes the LLM call and hands back a typed response, the same shape as
`litellm.messages()` in Python.

```rust
let response = litellm_core::messages::messages(MessagesRequest {
    model: "claude-sonnet-4-5",
    body,
    api_key: Some(key),
    ..
})
.await?;
```

Python continues to own configuration, retries, routing policy, logging,
callbacks, spend tracking, and customer plugins until each Rust path has parity
coverage and production evidence.

## Crates

| Crate | Role |
|-------|------|
| litellm-core | Provider transformations and shared types. OCR is deterministic transformation only. |
| litellm-runtime | Reusable OCR resolution, auth, preparation, lifecycle, HTTP, and polling. |
| litellm-ai-gateway | Axum and WebSocket host plus host-specific OCR logger/guardrail adapters. |
| litellm-python-bridge | PyO3 cdylib. Calls runtime directly for OCR and retains gateway for unrelated legacy paths. |

Dependency direction (acyclic): ai-gateway/python-bridge -> runtime -> core.

## Layout

```text
crates/
  core/           Shared types and provider transformations.
    src/messages/   Legacy whole-call route implementation.
    src/providers/anthropic/messages/transformation.rs
  runtime/        Reusable OCR execution.
  ai-gateway/     Axum server + WebSocket hosts; adapts OCR runtime hooks.
  python-bridge/  PyO3 bridge for Python LiteLLM.
```

The folder shape follows the Python provider tree:
`core/src/providers/<provider>/<route>/transformation.rs`. The bridge exposes one
function per top-level route, calling the matching core or runtime entrypoint.

## Checks

Run these before pushing Rust changes. GitHub Actions runs the same checks for
changes under `litellm-rust/`.

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
