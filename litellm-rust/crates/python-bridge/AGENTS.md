litellm-python-bridge is the PyO3 cdylib that exposes LiteLLM Rust APIs to the Python SDK. Keep API registration, domain dependency wiring, request assembly, and Python exception mapping here. Put domain-neutral Python/Serde conversion and GIL primitives in litellm-python-interop.

Keep it thin: no business logic, no transforms, no I/O orchestration — just marshal in/out and call the core entrypoint.
