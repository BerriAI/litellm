# Rust Provider Registry

`providers.json` is the source of truth for provider identity in `litellm-rust`.
`crates/core/build.rs` reads it at compile time and generates the typed
`LlmProvider` enum plus static provider metadata. Runtime code does not parse
this JSON.

To add a provider:

- Add a `providers.json` entry with `routing_name` matching Python
  `LlmProviders.value` in `litellm/types/utils.py`.
- Set `display_name` to the human-readable provider name for docs/errors.
- Set `default_api_base` to a stable provider-level default base URL, or `null`
  when it is unknown, dynamic, or route-specific.
- Set `api_key_env_var` to the canonical LiteLLM env var, or `null` when there
  is no single provider-level key.
- Put request/response logic under
  `crates/providers/src/<provider>/<route>/transformation.rs`; do not put
  transforms, signing logic, or secrets in this registry.
- Run `cargo test -p litellm-core --locked`; it verifies the Rust registry stays
  in parity with Python `LlmProviders`.
