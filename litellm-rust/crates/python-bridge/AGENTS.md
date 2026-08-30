litellm-python-bridge is the PyO3 cdylib that exposes Rust to the litellm Python SDK. It calls `litellm-runtime` directly for OCR and core entrypoints for routes such as Messages. The ai-gateway dependency remains for unrelated audio and WebSocket paths.

Keep it thin: no business logic, no transforms, no I/O orchestration. Marshal in/out and call the runtime or core entrypoint.
