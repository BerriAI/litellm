# HTTP

## Python surface today

- TBD: the Python HTTP clients and transports, and which of them users configure

## Shared crates

- `crates/http` owns outbound clients, TLS, proxies and timeouts, with `HttpClientPool` in `pool.rs`
- Rust routes take clients from `HttpClientPool` and never build their own per route

## Per host

- python-sdk and python-proxy
  - `python-bridge` injects the pool
  - Which async transport is in use and timeout conversion are host calls today (`http_transport`, `timeouts` in `litellm/rust_bridge/host/AGENTS.md`). TBD whether they stay host calls
- rust-proxy
  - Builds the pool from `config.yaml`

## Wire contracts

- TBD

## Order and gates

- TBD
