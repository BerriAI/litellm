# Rust Provider Metadata

The repo-root `provider_endpoints_support.json` is the shared source of truth
for provider identity and docs metadata in `litellm-rust`. `crates/core/build.rs`
reads it at compile time and generates the typed `LlmProvider` enum plus static
provider metadata. Runtime code does not parse this JSON.

To add a provider:

- Add a `provider_endpoints_support.json` provider entry using the LiteLLM
  provider slug, display name, docs URL, and endpoint support flags.
- Add optional defaults under the top-level `default_creds` map only when there
  is a stable provider-level base URL. Keep route-specific key env var names in
  the provider transform/config so auth resolution has one owner.
- Put request/response logic under
  `crates/providers/src/<provider>/<route>/transformation.rs`; do not put
  transforms, signing logic, or secrets in provider metadata.
- Run `cargo test -p litellm-core --locked`; it verifies the Rust registry stays
  in parity with `provider_endpoints_support.json`.
