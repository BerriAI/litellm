# Adding a provider / route to litellm-rust

Everything for a route lives in `crates/core/src/<route>/`; `crates/core/src/messages` is the reference. A host (the axum gateway, the Python bridge) only calls the route's entrypoint.

1. **Entrypoint** — `mod.rs`: `pub async fn <route>(request) -> CoreResult<Response>`, the Rust equivalent of `litellm.<route>()`, plus a `<route>_stream` variant when the route streams. It is the only thing a host touches.
2. **Transform contract** — `transformation.rs`: a `…ProviderConfig` trait (URL build + request/response transforms) with types in `types.rs`.
3. **Provider config** — `crates/core/src/providers/<provider>/<route>/transformation.rs`: implement that trait as a `const <PROVIDER>_<ROUTE>_CONFIG`, mirroring the Python provider tree. Add parity unit tests.
4. **Prepare + handler** — `prepare.rs` resolves provider/model, credentials, auth headers, and URL, then transforms the request; `handler.rs` performs the provider call through the shared client in `client.rs` and transforms the response.

## Coding standards

Before writing new logic, look for an existing base to extend. When a change is
“the same behavior for one more provider/endpoint/integration”, the codebase
almost always already has a shared abstraction for it (for example, provider
`BaseConfig` transformation classes in `litellm/llms/base_llm/`, shared
helpers in `litellm_core_utils/`, typed request/response models, or factory
functions). Find it first with a search, then add the new variant by inheriting
from or composing that base, overriding only what genuinely differs (model
name, parameter mapping, or auth).

Never copy an existing implementation and edit it in place, and never hand-roll
a parallel version of logic a base already provides. If you catch yourself
writing a second copy of a pattern that exists twice already, stop and extract a
base instead: put the shared shape in one place and make both call sites thin
variants of it. The test for a good abstraction is that adding the next provider
is a few declarative lines, not a new file of duplicated flow. Only diverge from
the base when behavior is genuinely different, and say so explicitly in the PR.

**Calling:** hosts invoke the core entrypoint — the Python bridge and the `ai-gateway` route service both call `litellm_core::messages::messages`. Never add a provider handler to `ai-gateway`. Register new modules in `lib.rs` / `mod.rs`, then run `cargo fmt && cargo clippy --workspace -- -D warnings && cargo test --workspace`.
