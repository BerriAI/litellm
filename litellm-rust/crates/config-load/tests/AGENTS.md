## Fixtures

- `corpus.rs` requires every config in `fixtures/valid/` to load and every config in `fixtures/invalid/` to fail. There is no known-gap list, so a valid config that fails to load is fixed in `src/`
- A config goes in `invalid/` only when Python rejects it too, either by raising on read or by crashing when the proxy loads it
- `#[files]` globs at compile time, so `build.rs` reruns the build when `fixtures/` changes
