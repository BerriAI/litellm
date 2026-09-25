# Rust workspace rules

## Test placement

- Never create a `tests.rs` (or `test.rs`) file under `src/`, and never `#[path = "tests.rs"] mod tests;`
- A test that reaches private items lives inline, in a `#[cfg(test)] mod tests { ... }` at the bottom of the file that owns those items
- A test that only uses the crate's public API lives in `crates/<crate>/tests/<subject>.rs`, next to `src/`
- Split a mixed test file along that line instead of widening visibility to move it
- A test for another crate's item belongs in that crate, not in a downstream one
- Never set `autotests = false` or hand-list `[[test]]` targets; every file directly under `tests/` is discovered by cargo, and a shared helper goes in `tests/<name>/mod.rs` or `tests/<subject>/support.rs` so it is not picked up as a test crate of its own

## Error definitions

- A crate's errors live in `src/error.rs`, defined with `thiserror`, and re-exported from `lib.rs`
- Default to one top-level `Error` enum per crate, with one variant per failure mode and a `#[error(...)]` message on each
- Wrap a lower-level error as a variant with `#[from]` or `#[source]` instead of flattening it to a string
- Exception: split into separate types when different functions fail in disjoint ways, especially when different callers see them. A shared enum would force every caller to match variants its function can never return
- Name a split type after what went wrong (a unit struct is fine for a single failure mode), not after the function that returns it
