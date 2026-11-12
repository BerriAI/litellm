# Rust host modules

Python behavior Rust needs but does not own, behind one module per question. Rust imports a module here by path and calls a function by name

## Boundary

- This package is the only place Rust reaches into Python. `python-bridge` and `callbacks-legacy-python` import no other LiteLLM module, and never do `py.import("litellm").getattr(...)` on a global
  - The one exception is `litellm.rust_bridge.execution`, the driver's step classes. It is runtime protocol, not Python state, and the Python entrypoints import it too
- Only Rust calls these modules. No Python module outside `host/` imports one. Tests may, to set up the Python state Rust reads
  - Logic both sides need lives with its Python owner (`public_call`, `secret_managers`, the route's types), and the host module calls it
- A Python call that reproduces Python behavior goes through a host function, including standard library and third-party calls: error text built from `httpx`, `OSError` or `litellm.exceptions`, `huggingface_hub` downloads, the process `random` generator
  - Rust reads the process environment itself (`ProcessEnvironment`). CPython's `os.environ` writes through to it, so there is nothing to ask Python for
  - Rust imports a module directly only to get a type or a Python builtin it applies itself: `isinstance` checks (`os.PathLike`, `ssl.SSLContext`), exception classes it raises (`redis.exceptions`), `builtins.object` and `inspect.getattr_static` for Python attribute semantics
  - `litellm-host-python` is the generic runtime and keeps its own `asyncio`, `contextvars` and `json` imports
- Check these invariants by review whenever a Rust file adds a `py.import`, an `import_exception!` or a `*_MODULE` constant, or a Python file imports from `litellm.rust_bridge.host`

## Modules

- One module answers one question
  - `model_capabilities`: what a model can do
  - `callbacks_legacy_python`: the legacy callback registries
  - `settings`: configured snapshots
  - `cache`: the live configured cache
  - `diagnostics`: legacy failure logging
  - `logger`: where native diagnostics are emitted
  - `preflight`: the SDK globals the native preflight borrows
  - `secret_manager`: the configured secret manager, its native config and the error text Python's managers produce
  - `http_transport`: which async transport is in use, and whether a call must send through Python's `AsyncHTTPHandler`
  - `timeouts`: timeout to seconds
  - `tokenizers`: bundled tiktoken files and Hugging Face tokenizer downloads
  - `randomness`: the process `random` generator the Python router draws from
  - `redis_events`: reporting native Redis calls and breaker transitions to `ServiceLogging`
  - `redis_credentials`: what a caller's redis-py `CredentialProvider` authenticates with
  - `model_cost`: where a `register_model` entry lands in the published table and the row it writes
- `routes/<route>.py` holds the questions one route asks, under the same function names on every route (`response`, `map_failure`, `connection_defaults`), so Rust names only the module. A question two routes ask goes in a top-level module here instead
- Each module reuses the Python implementation the Python route runs (`AnthropicModelInfo`, `Logging`). Do not re-derive those rules here or in Rust; one resolver, two callers

## Contract

- The module path and function signature are a contract. Each Rust crate declares a host module's path once, in a `const *_MODULE` next to the code that asks (`python-bridge/src/secrets/mod.rs`, `python-bridge/src/routes/<route>/`, `callbacks-legacy-python/src/python.rs`). Change both sides together and keep `tests/unit/rust_bridge/host/` green
- Inputs and outputs use values and frozen snapshots. Retain caller-owned clients and caches as opaque references when identity is required; Rust never discovers them through module globals
- Moving a question into Rust means replacing the function body with a native call, not adding a second resolver. The Python signature stays

## model_cost

- `model_cost` answers what Rust needs while registering models: `entry`, `published`, `write`, `resolve`, `merged` and `kept`. The live table, its version and every Python writer belong to `litellm_core_utils/model_cost_table.py`
- `resolve`, `merged` and `kept` are shims until `get_model_info` and a merge generic over Python values move to Rust. Then their bodies become native calls and the signatures stay
- `resolve` returns `None` when it was handed a table that is no longer published. That is how a pass learns a reload swapped the table
