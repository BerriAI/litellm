# Secret manager

## Python surface today

- A secret manager class users configure. It resolves secret references in config, such as `os.environ/...` and manager refs
- TBD: the exact classes, how users swap them, and which caching rules apply

## Shared crates

- `crates/secrets` and the `secrets-*` crates (AWS, Azure, Google, HashiCorp, CyberArk) resolve secrets
- `crates/secrets-types` holds the shared types

## Per host

- python-sdk and python-proxy
  - Same pattern as [cache](cache.md). The core asks through a host call (`secret_manager` in `litellm/rust_bridge/host/`), and the configured Python object answers
  - The built-in Python secret managers keep their API and become shims over the `secrets-*` crates
  - A user-supplied Python secret manager with no Rust equivalent is answered by Python on the SDK. TBD whether that is supported
- rust-proxy
  - Resolves secrets with the `secrets-*` crates from `config.yaml`

## Wire contracts

- TBD

## Order and gates

- TBD
