# Rust host modules

Python state Rust needs but does not own, behind one module per question. Rust imports a module here by path and calls a function by name

- One module answers one question: `provider` (which provider serves a model), `model_capabilities` (what a resolved model can do), `callbacks_legacy_python` (the legacy callback registries)
- A module is the only place Rust reads that Python state. Rust never does `py.import("litellm").getattr(...)` on a global; it asks here
- Each module reuses the Python implementation the Python route runs (`get_llm_provider`, `AnthropicModelInfo`, `Logging`). Do not re-derive those rules here or in Rust; one resolver, two callers
- The module path and function signature are a contract. The Rust side pins the path in a `const MODULE` in the crate that asks (`python-bridge/src/provider.rs`, `callbacks-legacy-python/src/python.rs`); change both sides together and keep `tests/unit/rust_bridge/host/` green
- Inputs are plain values (`str`, `None`, dicts) and outputs are plain values or `None`. No `litellm` objects cross the boundary
- Admission and execution ask the same function, so the provider that admits a call to Rust is the provider that serves it and names its errors. Resolve from the effective arguments after callbacks, never from a snapshot taken before them
- Moving a question into Rust means replacing the function body with a native call, not adding a second resolver. The Python signature stays
- A `*/route_host.py` module is not this: it projects one route's kwargs and maps that route's failures. Shared questions belong here
