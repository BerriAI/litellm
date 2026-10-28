---
name: rust-error
description: Define, wrap, convert, classify, report or test Rust error types in litellm-rust with thiserror. Use when adding an error type or variant, mapping one crate's error into another, turning an error into a gateway response or Python exception, deciding retry or fault from an error, or replacing call sites that build errors from strings
---

# Rust errors

`crates/http/src/error.rs` is the reference implementation. Read the reference for the part you are touching before editing, each one carries the reasoning and worked examples behind its rules

- [shape](references/shape.md)
  - One top-level `Error` per crate in `src/error.rs`. A public enum by default, an opaque struct over a private `Repr` enum when the variants must stay private. No `#[non_exhaustive]`
  - A variant is a failure mode a caller handles differently. Provider names and guidance are data, never variants
- [messages and causes](references/messages-and-causes.md)
  - Every message is written once, in `#[error(...)]`, and describes its own layer only. Call sites pass typed facts, never a string
  - The cause is the `source()`, never part of the message. Keep it typed, never stringify or drop it
  - `reqwest::Error` enters our types only through `litellm_http::Error`, which strips its URL
  - Anything shown to someone else is rendered with `litellm_error::Report`, never `to_string()`
- [conversions](references/conversions.md)
  - One `From` per source and target pair, only when the conversion needs no context. It classifies and sanitizes once
  - Nest a lower crate's error in one variant. Translate when the meaning changes, never mirror its variants
- [classification](references/classification.md)
  - Callers ask predicates named after their decision, they do not match nested variants. Predicates match exhaustively
  - Classification states facts. Retry and fallback policy lives with the caller that knows idempotency
- [boundaries](references/boundaries.md)
  - An error becomes text for someone else only in one mapping function per error type, in the gateway crates, the Python bridge and host-python. The inference gateway and the bridge decide by the same predicates
  - A failure before the provider was called is a decline. Both ask `declined()` first and hand it back to the host unchanged
- [gateway](references/gateway.md)
  - One `Error` with `IntoResponse` per gateway crate that serves a router. Every failure flows through the same `From` chain, is classified once in `status()` and rendered by surface and framing
- [pyo3](references/pyo3.md)
  - Two `create_exception!` classes with the payload in `args`, `ValueError` for the caller's fault and `RuntimeError` otherwise, decoded in `failures.py`
- [testing](references/testing.md)
  - Test predicates, chain shape and redaction on errors built the way production builds them
- [recipes](references/recipes.md)
  - Common tasks walked through end to end

The workspace uses thiserror 2 for every error type, and no library crate uses `anyhow` or `Box<dyn Error>` in a public signature. The [derive docs](https://docs.rs/thiserror/latest/thiserror/) are the authority on macro behavior, and the [`std::error::Error` docs](https://doc.rust-lang.org/std/error/trait.Error.html) on what `Display` and `source()` mean
