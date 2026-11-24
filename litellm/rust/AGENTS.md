# Rust workspace rules

For keeping secrets out of Debug output, follow [.agents/skills/rust-redaction/SKILL.md](.agents/skills/rust-redaction/SKILL.md)

For diagnostic tracing changes, follow [.agents/skills/rust-tracing/SKILL.md](.agents/skills/rust-tracing/SKILL.md)

For string-valued enums and their Serde conversions, follow [.agents/skills/rust-string-enums/SKILL.md](.agents/skills/rust-string-enums/SKILL.md)

For error types, error variants and conversions between crate errors, follow [.agents/skills/rust-error/SKILL.md](.agents/skills/rust-error/SKILL.md)

For where config.yaml values, environment layering and kwargs from Python become typed Rust, follow [.agents/skills/rust-python-boundary/SKILL.md](.agents/skills/rust-python-boundary/SKILL.md)

The process-wide rustls `CryptoProvider` (ring) is installed once by the host. rust-proxy does it at the top of `main` through `litellm_gateway::install_crypto_provider`, and python-bridge must do the same in its module init before it builds any rustls config. Library crates never install it. The AWS SDK links aws-lc-rs next to ring, so without an installed default the first rustls config a library builds, such as redis-rs TLS, panics

For fieldless enums, derive `strum::VariantArray` and use `VARIANTS` instead of a hand-listed `ALL` array. Derive Strum string conversions instead of hand-written variant-to-string matches

Use the derived conversions directly (`<&'static str>::from(x)` / `.into()`, `str::parse`) with no `as_str`/`parse` wrapper that only delegates, and spell each variant with explicit `#[strum(serialize = "...")]` instead of `serialize_all`

## Porting from Python

- Put all Python compatibility (truthiness, either-of lookups, Python `str()` formatting and similar) in the `python-compat` crate, never inline in the crate being ported
- Do not keep the old Python implementation around to run parity tests against. If a parity check is truly needed, capture the Python outputs once as JSON fixtures
- Do not match Python error messages exactly. Never run a Python replay or a host call just to produce an error
- Parse into typed structs at the boundary and keep `serde_json::Value` out of internal signatures
- Keep logic out of the Python side of the host. If a host call is small, port it to Rust instead
- Keep the host interface as small as possible
- Do not reproduce Python string and parsing quirks around odd values such as huge integers or unusual characters
- Add a doc comment to every file and struct that holds significant logic
- Say "Rust", not "Native", when naming something written in Rust

## Test placement

- Never create a `tests.rs` (or `test.rs`) file under `src/`, and never `#[path = "tests.rs"] mod tests;`
- Tests live inline by default, in a `#[cfg(test)] mod tests { ... }` at the bottom of the file that owns the code under test, whether or not they touch private items
- `crates/<crate>/tests/` is only for tests that drive several modules together end to end, such as a full request through a route. A test of one type or function never goes there
- A helper shared by several inline test modules goes in a `#[cfg(test)] mod test_support;` at the crate root, and only once a second module needs it
- A test for another crate's item belongs in that crate, not in a downstream one
- A fake or helper that other crates' tests need goes behind `#[cfg(any(test, feature = "test-support"))]` in the crate that owns it, with a `test-support` feature in its `Cargo.toml`. Downstream crates turn the feature on only under `[dev-dependencies]`, or from their own `test-support` feature, so it never reaches a shipped build
- Tests that call a real provider go in `crates/<crate>/tests/live/<provider>.rs` behind a `tests/live/main.rs`. Each reads the provider's key from the environment through an `Option` fixture and returns early when it is unset, never `#[ignore]`, so `cargo test` stays offline until the key is exported
- Never set `autotests = false` or hand-list `[[test]]` targets. Every file directly under `tests/` is discovered by cargo, and a shared helper goes in `tests/<name>/mod.rs` or `tests/<subject>/support.rs` so it is not picked up as a test crate of its own

## Test fixtures and cases

Use [`#[rstest]`](https://docs.rs/rstest/latest/rstest/attr.rstest.html) for new and updated tests and [`#[fixture]`](https://docs.rs/rstest/latest/rstest/attr.fixture.html) for reusable setup, injected through typed test arguments. Express input variations as named `#[case::name(...)]` cases instead of loops or duplicated tests so each failure identifies its case. Keep behavior assertions in the test body and fixtures focused on setup. Use the workspace `rstest` dependency

- Never loop over inputs (`for`, `.iter().for_each`, `.all`) inside a test body. Give each input its own `#[case::name(...)]`, or use `#[values(...)]` for a cross product
- Exception: a test pinning a Rust table against a repo-owned data file (for example `include_str!` of a JSON config) may iterate that file's entries
