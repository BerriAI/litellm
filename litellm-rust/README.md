# LiteLLM Rust

The Cargo workspace and lockfile live at the repository root. Rust sources live under `litellm-rust/`

## Packages

| Package | Location | Responsibility |
|---------|----------|----------------|
| `litellm` | `apps/litellm` | CLI that starts the gateway |
| `litellm-core` | `crates/core` | Provider calls, transforms, routing, and callback contracts |
| `litellm-ai-gateway` | `crates/ai-gateway` | HTTP/WebSocket server, client authentication, configuration, and callback I/O |
| `litellm-python-bridge` | `crates/python-bridge` | PyO3 adapter exposing core calls to Python |

The CLI depends on the gateway. The gateway and Python bridge each depend on core, independently of one another

## Run

From the repository root, with LiteLLM importable in the Python interpreter used by PyO3:

```bash
cargo run -p litellm -- --config litellm-rust/crates/ai-gateway/config.yaml
```

`--config` overrides `LITELLM_CONFIG_PATH`. With neither set, the server uses its existing environment-based deployment fallback

The `litellm-ai-gateway` binary remains available. It runs without Python by default; enable `python-config` to load a proxy configuration file

See the [gateway README](crates/ai-gateway/README.md) for authentication, environment variables, and deployment configuration

## Checks

Run from the repository root. `rust-toolchain.toml` pins the compiler and tools

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo clippy -p litellm-core --all-targets --no-default-features --locked -- -D warnings
cargo test -p litellm-core --no-default-features --locked
```

The workspace tests enforce the package set and dependency direction. The [Realtime benchmark](crates/ai-gateway/benches/realtime/README.md) runs separately and requires provider credentials
