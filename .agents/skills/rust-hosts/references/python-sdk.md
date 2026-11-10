# python-sdk on the Rust core

`import litellm` in a user's process (`litellm.completion`, `Router`, `litellm.Cache`). It is a permanent host. It runs the shared crates through the PyO3 extension, `crates/python-bridge`

## What reaches Rust

- Values arrive as Python already resolved them, such as the kwargs `RedisCache` was built with. The bridge converts them and hands them to the same shared resolver rust-proxy feeds from `config.yaml` (`RedisSettings::resolve` in `crates/config`)
- `litellm.*` globals such as `litellm.drop_params` are read through a host call, because users set them at runtime and Python owns them
- Rust exposes a shared-crate operation as a pyclass. Python never sees the underlying protocol, for example one pyclass per Redis keyspace and no Redis commands
- Results become Python values and errors map to the existing public exceptions. Signatures of public Python entry points stay the same

## What only this host can do

- Rust may call Python through `litellm/rust_bridge/host/`, one module per question, under the contract in that directory's `AGENTS.md`
- That covers classes users configure or swap (`litellm.cache`, the secret manager) and Python-owned state such as `litellm.*` settings
- The built-in classes behind those calls are shims over shared crates, so a host call that lands in Rust again is expected
- A user-supplied Python object with no Rust equivalent (`credential_provider`) stays an SDK-only feature, with a clear error where it is unsupported

## Where the rules live

- `litellm/rust_bridge/AGENTS.md` covers choosing Rust or Python per call and falling back to Python
- No product rule lives in `python-bridge`. A rule the extension needs that rust-proxy also needs moves into a shared crate
