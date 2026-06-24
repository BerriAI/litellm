# Adding a provider / route to litellm-rust

Three layers, same for every route (see `ocr` and `realtime` as references):

1. **Transform contract (pure)** — `crates/core/src/<route>/transformation.rs`: a `…ProviderConfig` trait (URL build + request/response transforms) + types in `types.rs`. No network, env, or auth.
2. **Provider config (pure)** — `crates/providers/src/<provider>/<route>/transformation.rs`: implement that trait as a `const <PROVIDER>_<ROUTE>_CONFIG`, mirroring the Python provider tree. Add parity unit tests.
3. **HTTP / transport (the host)** — `crates/providers/src/<route>.rs` (e.g. `ocr.rs`, `realtime.rs`): the callable fn (`run_ocr`, `realtime`). It resolves the key, builds the auth header, builds URL + transforms via the config, then does the network call. This is the only layer allowed to do I/O.

**Calling:** the host invokes the route fn — the Python bridge calls `run_ocr`; the `ai-gateway` server calls `realtime`. Register new modules in `lib.rs` / `mod.rs`, then run `cargo fmt && cargo clippy --workspace -- -D warnings && cargo test --workspace`.
