# Rust workspace rules

For adding or editing AGENTS.md files and skills, follow [.agents/skills/agent-docs/SKILL.md](.agents/skills/agent-docs/SKILL.md)

For diagnostic tracing changes, follow [.agents/skills/rust-tracing/SKILL.md](.agents/skills/rust-tracing/SKILL.md)

For string-valued enums and their Serde conversions, follow [.agents/skills/rust-string-enums/SKILL.md](.agents/skills/rust-string-enums/SKILL.md)

For response-cache keys, storage selection, and the cache service contract, follow [.agents/skills/response-cache/SKILL.md](.agents/skills/response-cache/SKILL.md)

## Test placement

- Never create a `tests.rs` (or `test.rs`) file under `src/`, and never `#[path = "tests.rs"] mod tests;`
- A test that reaches private items lives inline, in a `#[cfg(test)] mod tests { ... }` at the bottom of the file that owns those items
- A test that only uses the crate's public API lives in `crates/<crate>/tests/<subject>.rs`, next to `src/`
- Split a mixed test file along that line instead of widening visibility to move it
- A test for another crate's item belongs in that crate, not in a downstream one
- Never set `autotests = false` or hand-list `[[test]]` targets; every file directly under `tests/` is discovered by cargo, and a shared helper goes in `tests/<name>/mod.rs` or `tests/<subject>/support.rs` so it is not picked up as a test crate of its own

## Test fixtures and cases

Use [`#[rstest]`](https://docs.rs/rstest/latest/rstest/attr.rstest.html) for new and updated tests and [`#[fixture]`](https://docs.rs/rstest/latest/rstest/attr.fixture.html) for reusable setup, injected through typed test arguments. Express input variations as named `#[case::name(...)]` cases instead of loops or duplicated tests so each failure identifies its case. Keep behavior assertions in the test body and fixtures focused on setup. Use the workspace `rstest` dependency

## Error definitions

- A crate's errors live in `src/error.rs`, defined with `thiserror`, and re-exported from `lib.rs`
- Put message templates in the variant's `#[error(...)]` declaration. Callers pass only the small typed arguments needed to fill them, never `Error::Variant(format!(...))` or a preformatted message. Keep the smallest set of neutral variants that callers need to distinguish; different wording or providers do not justify new variants
- Default to one top-level `Error` enum per crate, with one variant per failure mode and a `#[error(...)]` message on each. A failure mode is something a caller handles differently (phase, status code, retry, a message Python parity pins exactly); failures no caller tells apart share one variant and differ only in its message
- Keep shared error enums minimal and provider-neutral. Provider names, credential types, configuration fields, and setup guidance belong in caller-supplied data, not dedicated variants or hardcoded shared messages. Reuse a variant for the same failure mode across providers, such as `MissingApiBase { provider: "Azure", guidance: "..." }`. An exact parity message does not justify a provider-specific variant when caller-supplied context can preserve it
- Wrap a lower-level error as a variant with `#[from]` or `#[source]` instead of flattening it to a string
- Exception: split into separate types when different functions fail in disjoint ways, especially when different callers see them. A shared enum would force every caller to match variants its function can never return
- Name a split type after what went wrong (a unit struct is fine for a single failure mode), not after the function that returns it
