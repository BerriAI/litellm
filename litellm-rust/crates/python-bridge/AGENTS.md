litellm-python-bridge is the PyO3 cdylib that exposes Rust to the litellm Python SDK — a thin adapter (Python objects → Rust calls → Python results) over litellm-ai-gateway.

Keep it thin: no business logic, no transforms, no I/O orchestration — just marshal in/out and call into litellm-ai-gateway.
