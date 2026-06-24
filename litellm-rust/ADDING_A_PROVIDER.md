# Adding a provider / route to litellm-rust

- Keep the route contract pure in `crates/core/src/<route>/`: define the typed request/response structs and a provider config trait with no network, env, auth, or logging.
- Put provider-specific transforms in `crates/providers/src/<provider>/<route>/transformation.rs`, mirroring the Python provider tree and exposing a `const <PROVIDER>_<ROUTE>_CONFIG`.
- The provider config owns three pure steps: map LiteLLM params, transform the LiteLLM request into the provider request, and transform the provider response back into the LiteLLM response.
- If the provider has a reverse or normalization step, keep it pure and explicit next to the transforms; do not hide reverse mapping inside the HTTP transport.
- Route host functions in `crates/providers/src/<route>.rs` must be async: resolve auth/base URL, call the transforms, send with async transport, then call the response transform.
- Register modules in `lib.rs` / `mod.rs`, add parity tests for params/request/response behavior, then run `cargo fmt && cargo clippy --workspace --all-targets --locked -- -D warnings && cargo test --workspace --locked`.
