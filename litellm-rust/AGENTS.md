# AGENTS.md

The repository-root Cargo workspace has three library/host crates under `litellm-rust/crates` and the `litellm` CLI under `litellm-rust/apps`. Routes and providers are modules within those layers

## Crates

| Crate | Role |
|-------|------|
| litellm-core | The LiteLLM SDK in Rust. One public entrypoint per top-level call (`messages::messages()`), owning types, transforms, provider resolution, auth, the provider HTTP call, and the callback/guardrail contracts. Call it, get a typed response. |
| litellm-ai-gateway | The axum server host. Owns transport, client authentication, config, and socket splicing; delegates provider calls to core. |
| litellm-python-bridge | PyO3 cdylib exposing core calls to the Python SDK. |
| litellm | CLI entrypoint that starts the gateway. |

The CLI depends on the gateway. The gateway and Python bridge each depend on core; the Python bridge must not depend on the gateway

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

Handlers never live in `ai-gateway`. The one host-side exception is the realtime WebSocket transport (dial, warm pool, splice) under `ai-gateway/src/routes/realtime/`; its event transforms live in `core` (`realtime`, `providers/openai/realtime`).

Add a crate only for a separate artifact or compilation boundary. New providers and routes belong in modules

When adding a package, update the root workspace members, `crates/core/tests/workspace_crate_allowlist.rs`, and this file

## Style

All Rust in `litellm-rust/` follows the official Rust Style Guide:
[Rust Style Guide](https://doc.rust-lang.org/style-guide/)

`rustfmt` implements its formatting by default, so run `cargo fmt` before committing; CI gates every PR on `cargo fmt --check`. Do not hand-format against rustfmt or add a `rustfmt.toml` that diverges from the default style.

Use `snake_case` for functions/modules, `UpperCamelCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Follow the guide's import grouping and item ordering

Run the checks listed in README.md from the repository root. Keep provider behavior covered by regression and parity tests; moving code between layers must preserve the Python interface
