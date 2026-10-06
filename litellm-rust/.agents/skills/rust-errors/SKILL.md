---
name: rust-errors
description: Define, split, or change Rust error types in litellm-rust, including where they live, how variants map to failure modes, and how messages are templated
---

# Rust errors

- A crate's errors live in `src/error.rs`, defined with `thiserror`, and re-exported from `lib.rs`
- Put message templates in the variant's `#[error(...)]` declaration. Callers pass only the small typed arguments needed to fill them, never `Error::Variant(format!(...))` or a preformatted message. Keep the smallest set of neutral variants that callers need to distinguish; different wording or providers do not justify new variants
- Default to one top-level `Error` enum per crate, with one variant per failure mode and a `#[error(...)]` message on each. A failure mode is something a caller handles differently (phase, status code, retry, a message Python parity pins exactly); failures no caller tells apart share one variant and differ only in its message
- Keep shared error enums minimal and provider-neutral. Provider names, credential types, configuration fields, and setup guidance belong in caller-supplied data, not dedicated variants or hardcoded shared messages. Reuse a variant for the same failure mode across providers, such as `MissingApiBase { provider: "Azure", guidance: "..." }`. An exact parity message does not justify a provider-specific variant when caller-supplied context can preserve it
- Wrap a lower-level error as a variant with `#[from]` or `#[source]` instead of flattening it to a string
- Exception: split into separate types when different functions fail in disjoint ways, especially when different callers see them. A shared enum would force every caller to match variants its function can never return
- Name a split type after what went wrong (a unit struct is fine for a single failure mode), not after the function that returns it
