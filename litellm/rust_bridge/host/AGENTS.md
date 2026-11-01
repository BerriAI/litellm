# Rust host modules

Python state Rust needs but does not own, behind one module per question. Rust imports a module here by path and calls a function by name

- One module answers one question: `model_capabilities` (what a model can do), `callbacks_legacy_python` (the legacy callback registries), `settings` (configured snapshots), `cache` (the live configured cache), `diagnostics` (legacy failure logging), `secret_manager` (the configured secret manager and native config capture), `http_transport` (which async transport is in use), `timeouts` (timeout to seconds)
- A module is the only place Rust reads that Python state. Rust never does `py.import("litellm").getattr(...)` on a global; it asks here
- Product-state questions are limited to a class users configure or swap (`cache`, `secret_manager`) or Python-owned values (`settings`). The test is the "Classes and values" rule in the [`rust-hosts`](../../../.agents/skills/rust-hosts/SKILL.md) skill
- Python-only compatibility glue (`diagnostics`, route response construction and exception mapping) also stays here. It preserves Python's public API, never shared product rules
- Rules and data rust-proxy also needs live in a shared crate, never here. One resolver, two callers, and that resolver is the shared crate. `model_capabilities` goes away once the native model catalog answers it
- The module path and function signature are a contract. The Rust side pins the path in a `const MODULE` in the crate that asks (`python-bridge/src/routes/messages/host.rs`, `python-bridge/src/secrets/`, `callbacks-legacy-python/src/python.rs`); change both sides together and keep `tests/unit/rust_bridge/host/` green
- Inputs and outputs use values and frozen snapshots. Retain caller-owned clients and caches as opaque references when identity is required; Rust never discovers them through module globals
- Moving a question into Rust means replacing the function body with a native call, not adding a second resolver. The Python signature stays
- `routes/<route>.py` holds the questions one route asks, under the same function names on every route (`response`, `map_failure`, `connection_defaults`), so Rust names only the module. A question two routes ask goes in a top-level module here instead
