# rust-proxy on the Rust core

The `gateway` crate tree. It links the shared crates directly and has no interpreter. Routes, extractors and envelopes follow the [`rust-gateway`](../../../../litellm-rust/.agents/skills/rust-gateway/SKILL.md) skill

## In front of python-proxy

- rust-proxy takes every request. It serves what it supports end to end in Rust and forwards the rest to python-proxy workers
- The decision is made once per request, before any limit or budget is charged, so nothing is counted twice. A forwarded request runs entirely in Python
- The forward list (routes, providers, config callbacks, unknown headers and body fields) shrinks as Rust supports more. The end state is rust-proxy alone

## Everything exists in Rust

- rust-proxy can't reach `litellm/rust_bridge/host/` or any Python state, unlike the extension embedded in Python. It also doesn't want to, so the end-to-end path stays fast
- It reads `config.yaml` and the database itself and resolves secrets with the `secrets-*` crates, then feeds the same shared resolvers the extension feeds with Python values
- It answers every host call natively through `host-native`, and supplies its own implementation of every injected trait

## Matching python-proxy

- Anything python-proxy writes that rust-proxy also writes (Redis keys and values, DB rows) is a wire contract, see [python-proxy](python-proxy.md)
