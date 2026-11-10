# Rust host modules

Python state Rust needs but does not own, behind one module per question. Rust imports a module here by path and calls a function by name

- One module answers one question: `model_capabilities` (what a model can do), `callbacks_legacy_python` (the legacy callback registries), `settings` (configured snapshots), `cache` (the live configured cache), `diagnostics` (legacy failure logging), `secret_manager` (the configured secret manager and native config capture), `http_transport` (which async transport is in use), `timeouts` (timeout to seconds), `redis_events` (reporting native Redis calls and breaker transitions to `ServiceLogging`), `redis_credentials` (what a caller's redis-py `CredentialProvider` authenticates with), `model_cost` (the live `litellm.model_cost` table, every write to it and its version)
- A module is the only place Rust reads that Python state. Rust never does `py.import("litellm").getattr(...)` on a global; it asks here
- Each module reuses the Python implementation the Python route runs (`AnthropicModelInfo`, `Logging`). Do not re-derive those rules here or in Rust; one resolver, two callers
- The module path and function signature are a contract. The Rust side pins the path in a `const MODULE` in the crate that asks (`python-bridge/src/routes/messages/host.rs`, `python-bridge/src/secrets/`, `callbacks-legacy-python/src/python.rs`); change both sides together and keep `tests/unit/rust_bridge/host/` green
- Inputs and outputs use values and frozen snapshots. Retain caller-owned clients and caches as opaque references when identity is required; Rust never discovers them through module globals
- Moving a question into Rust means replacing the function body with a native call, not adding a second resolver. The Python signature stays
- `routes/<route>.py` holds the questions one route asks, under the same function names on every route (`response`, `map_failure`, `connection_defaults`), so Rust names only the module. A question two routes ask goes in a top-level module here instead

## model_cost

- `publish`, `write` and `changed_in_place` are the only writers. Each moves `ModelCostVersion`
- The version is `(id(table), len(table), writes)`, so replacing the dict or changing its size moves it without the host
  - An in-place value edit does not, so code that edits rows in place calls `changed_in_place()`
- Anything derived from the table is cached under the version (`@VersionCached`, `model_cost_version()`), never cleared by hand
- `resolve`, `merged` and `kept` are shims until `get_model_info` and a merge generic over Python values move to Rust. Then their bodies become native calls and the signatures stay
- `resolve` returns `None` when it was handed a table that is no longer published. That is how a pass learns a reload swapped the table
- Compat names exist only so code written against main keeps working. New code never calls them, and they never gain behavior of their own
  - `utils._invalidate_model_cost_lowercase_map` forwards to `changed_in_place`
  - `utils.get_model_cost_mutation_generation` forwards to `model_cost_version().writes`
  - `utils._model_cost_lowercase_map` is always `None`
