# LiteLLM Rust

This workspace contains the staged Rust implementation for LiteLLM.

Rust starts as a pure transform core used by the existing Python host. Python
continues to own auth, configuration, network I/O, retries, routing, logging,
callbacks, spend tracking, and customer plugins until each Rust path has parity
coverage and production evidence.

## Crates

| Crate | Role | Pure / I/O |
|-------|------|------------|
| litellm-core | Translation layer — types, route contracts (traits), provider transforms (modules under providers/), and the router. Builds requests/responses; no network. | Pure |
| litellm-ai-gateway | Routes + host — the only crate that touches the network. HTTP/WebSocket I/O (modules under io/) plus the axum server binary (behind the `server` feature). | I/O |
| litellm-python-bridge | PyO3 cdylib exposing Rust to the litellm Python SDK — a thin adapter over litellm-ai-gateway's I/O. | Binding |

Dependency direction (acyclic): litellm-core ← litellm-ai-gateway ← litellm-python-bridge.

## Layout

```text
crates/
  core/           Route contracts, shared pure types, errors, and templates.
    src/ocr/
  providers/      Provider-specific pure transforms.
    src/mistral/ocr/transformation.rs
  python-bridge/  PyO3 bridge for Python LiteLLM.
```

The folder shape should follow the Python provider tree:
`providers/src/<provider>/<route>/transformation.rs`. The bridge should expose
one function per top-level route, starting with `ocr(payload)`.

## Checks

Run these before pushing Rust changes. GitHub Actions runs the same checks for
changes under `litellm-rust/`.

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
