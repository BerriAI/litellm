---
name: rust-testing
description: Write, move, or restructure Rust tests in litellm-rust, including where a test file goes and how to express fixtures and cases with rstest
---

# Rust testing

## Placement

- Never create a `tests.rs` (or `test.rs`) file under `src/`, and never `#[path = "tests.rs"] mod tests;`
- A test that reaches private items lives inline, in a `#[cfg(test)] mod tests { ... }` at the bottom of the file that owns those items
- A test that only uses the crate's public API lives in `crates/<crate>/tests/<subject>.rs`, next to `src/`
- Split a mixed test file along that line instead of widening visibility to move it
- A test for another crate's item belongs in that crate, not in a downstream one
- Never set `autotests = false` or hand-list `[[test]]` targets; every file directly under `tests/` is discovered by cargo, and a shared helper goes in `tests/<name>/mod.rs` or `tests/<subject>/support.rs` so it is not picked up as a test crate of its own

## Fixtures and cases

Use [`#[rstest]`](https://docs.rs/rstest/latest/rstest/attr.rstest.html) for new and updated tests and [`#[fixture]`](https://docs.rs/rstest/latest/rstest/attr.fixture.html) for reusable setup, injected through typed test arguments. Express input variations as named `#[case::name(...)]` cases instead of loops or duplicated tests so each failure identifies its case. Keep behavior assertions in the test body and fixtures focused on setup. Use the workspace `rstest` dependency
