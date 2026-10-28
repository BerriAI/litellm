# Rust workspace rules

For diagnostic tracing changes, follow [.agents/skills/rust-tracing/SKILL.md](.agents/skills/rust-tracing/SKILL.md)

For string-valued enums and their Serde conversions, follow [.agents/skills/rust-string-enums/SKILL.md](.agents/skills/rust-string-enums/SKILL.md)

For error types, error variants and conversions between crate errors, follow [.agents/skills/rust-error/SKILL.md](.agents/skills/rust-error/SKILL.md)

For fieldless enums, derive `strum::VariantArray` and use `VARIANTS` instead of a hand-listed `ALL` array. Derive Strum string conversions instead of hand-written variant-to-string matches

Use the derived conversions directly (`<&'static str>::from(x)` / `.into()`, `str::parse`) with no `as_str`/`parse` wrapper that only delegates, and spell each variant with explicit `#[strum(serialize = "...")]` instead of `serialize_all`

## Test placement

- Never create a `tests.rs` (or `test.rs`) file under `src/`, and never `#[path = "tests.rs"] mod tests;`
- Tests live inline by default, in a `#[cfg(test)] mod tests { ... }` at the bottom of the file that owns the code under test, whether or not they touch private items
- `crates/<crate>/tests/` is only for tests that drive several modules together end to end, such as a full request through a route. A test of one type or function never goes there
- A helper shared by several inline test modules goes in a `#[cfg(test)] mod test_support;` at the crate root, and only once a second module needs it
- A test for another crate's item belongs in that crate, not in a downstream one
- Tests that call a real provider go in `crates/<crate>/tests/live/<provider>.rs` behind a `tests/live/main.rs`. Each reads the provider's key from the environment through an `Option` fixture and returns early when it is unset, never `#[ignore]`, so `cargo test` stays offline until the key is exported
- Never set `autotests = false` or hand-list `[[test]]` targets. Every file directly under `tests/` is discovered by cargo, and a shared helper goes in `tests/<name>/mod.rs` or `tests/<subject>/support.rs` so it is not picked up as a test crate of its own

## Test fixtures and cases

Use [`#[rstest]`](https://docs.rs/rstest/latest/rstest/attr.rstest.html) for new and updated tests and [`#[fixture]`](https://docs.rs/rstest/latest/rstest/attr.fixture.html) for reusable setup, injected through typed test arguments. Express input variations as named `#[case::name(...)]` cases instead of loops or duplicated tests so each failure identifies its case. Keep behavior assertions in the test body and fixtures focused on setup. Use the workspace `rstest` dependency

- Never loop over inputs (`for`, `.iter().for_each`, `.all`) inside a test body. Give each input its own `#[case::name(...)]`, or use `#[values(...)]` for a cross product
- Exception: a test pinning a Rust table against a repo-owned data file (for example `include_str!` of a JSON config) may iterate that file's entries
