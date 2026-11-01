# python-proxy on the Rust core

The FastAPI proxy. It reaches Rust through the same extension as python-sdk, so everything in [python-sdk](python-sdk.md) applies. For a long time it runs behind rust-proxy and serves every request rust-proxy forwards to it. It is retired once rust-proxy supports everything

## What gets ported

- Code only python-proxy uses stays in Python. It serves the long tail until rust-proxy supports that request
- When rust-proxy needs the same behavior, write it in a shared crate for rust-proxy and leave python-proxy's code in place, with wire parity tests between the two
- Code python-sdk also uses (response cache, router cooldowns, affinity, router usage) goes through the shared crate from both Python hosts

## Shared state

- python-proxy and rust-proxy run against one Redis and one database, both in the same pod (rust-proxy forwarding to local Python workers) and across pods
- Every Redis key and value and every DB row both sides write is a wire contract. See [redis](redis.md) and [database](database.md)
- A behavior change to a shared key or row lands on both sides in the same change
