litellm-python-bridge is the PyO3 cdylib that exposes Rust to the litellm Python SDK — a thin adapter (Python objects → Rust calls → Python results) over the litellm-core route entrypoints (e.g. `litellm_core::messages::messages`).

Keep it thin: no business logic, no transforms, no I/O orchestration — just marshal in/out and call the core entrypoint.
