---
name: rust-error
description: Define, wrap, convert, classify, report or test Rust error types in litellm-rust with thiserror. Use when adding an error type or variant, mapping one crate's error into another, turning an error into a gateway response or Python exception, deciding retry or fault from an error, or replacing call sites that build errors from strings
---

# Rust errors

`crates/http/src/error.rs` is the reference implementation. Read the reference for the part you are touching before editing, each one carries the reasoning and worked examples behind its rules

- [shape](references/shape.md)
  - One top-level `Error` per crate in `src/error.rs`. A public enum by default, an opaque struct over a private `Repr` enum when the variants must stay private
  - A variant is a failure mode a caller handles differently. Provider names and guidance are data, never variants
- [messages and causes](references/messages-and-causes.md)
  - Every message is written once, in `#[error(...)]`. Call sites pass typed facts, never a string
  - Store the cause itself, never its string, and render it once
  - Strip `reqwest::Error` URLs where they enter our types
- [conversions](references/conversions.md)
  - One `From` per source and target pair, only when the conversion needs no context. It classifies and sanitizes once
  - Nest a lower crate's error in one variant. Translate when the meaning changes, never mirror its variants
- [classification](references/classification.md)
  - Callers ask predicates named after their decision, they do not match nested variants
  - Classification states facts. Retry and fallback policy lives with the caller that knows idempotency
- [boundaries](references/boundaries.md)
  - The gateway and the Python bridge are the only places an error becomes text for someone else
- [testing](references/testing.md)
  - Test predicates, chain shape and redaction on errors built the way production builds them
- [recipes](references/recipes.md)
  - Common tasks walked through end to end

The workspace uses thiserror 2. The [derive docs](https://docs.rs/thiserror/latest/thiserror/) are the authority on macro behavior
