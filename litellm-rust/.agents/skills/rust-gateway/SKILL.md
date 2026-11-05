---
name: rust-gateway
description: Add or change an HTTP route, extractor, response envelope, stream or router mount in the litellm-rust gateway crates (crates/gateway and crates/gateway-*). Use when adding an endpoint, parsing a request body, choosing an error envelope for an API surface, relaying SSE, or writing a router-level test
metadata:
  internal: true
---

# Rust gateway

`crates/gateway` mounts endpoint routers and serves the listener. `crates/gateway-inference`, `gateway-mcp` and `gateway-ui` own their endpoints, `gateway-auth` is the inbound authentication middleware and `gateway-management` holds management logic that has no router yet. A crate's `AGENTS.md`, where it has one, says what belongs in it. Read the reference for the part you are touching

- [surfaces](references/surfaces.md)
  - Each API surface mirrors the envelope the Python proxy returns there, so the shapes differ on purpose
  - The error type string is derived from the status and provider bodies pass through bounded
- [requests](references/requests.md)
  - Extractors reject with the crate's `Error`, handlers take the body as a `Result` when the error needs the request id
  - Every alias of a path is registered together, a known but unserved path goes to one handler
  - Streams relay through `litellm_host_http::Sse`, with the error frame rendered by the crate's `Error`
- [testing](references/testing.md)
  - Build the router through `tests/support` and drive it with `oneshot`, assert the fields the surface promises

How an error is classified before it reaches a response is the [`rust-error`](../rust-error/SKILL.md) skill, in particular its [gateway](../rust-error/references/gateway.md) page. Axum's own conventions are in its [docs](https://docs.rs/axum/latest/axum/)
