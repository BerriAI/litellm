litellm-python-bridge is the PyO3 cdylib exposing LiteLLM Rust APIs to the Python SDK.

- Owns API registration, domain dependency wiring, request assembly, and exception mapping
- Keep it thin: no business logic, no transforms, no I/O orchestration; just marshal in/out and call the core entrypoint
- One stable method per top-level route (`ocr`/`aocr`, `messages`/`amessages`, ...); do not add per-provider helpers
- Provider dispatch lives in `litellm-core`, never here
- Put domain-neutral Python/Serde conversion, retained callbacks, and Python↔Tokio execution in `litellm-python-interop`
- Two logical parts, kept as modules: the domain adapter (`src/routes/*`, `marshal`, `errors`) and the binding artifact (`#[pymodule]`, `#[pyfunction]`, registration in `lib.rs`)
- Data handling: do not log OCR payloads or provider responses; avoid copying large payloads; sanitize errors before they cross the boundary
- Tests: `cargo test --workspace` compiles here; Python tests cover disabled, enabled, and module-missing fallback for every exposed route
