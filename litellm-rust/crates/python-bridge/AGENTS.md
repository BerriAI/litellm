litellm-python-bridge is the PyO3 cdylib that exposes LiteLLM Rust APIs to the Python SDK. Keep API registration, domain dependency wiring, request assembly, and Python exception mapping here. Put domain-neutral Python/Serde conversion and GIL primitives in litellm-python-interop.

Keep it thin: no business logic, no transforms, no I/O orchestration — just marshal in/out and call the core entrypoint.

The retained routes (`ocr_retained`, and any future `retained_http` variants) are a Python-compatibility adapter. Python owns prepare, transform, and logging; Rust owns only the buffered POST and the boundary call marshaling. There is no retry, billing, guardrail, or logging orchestration here, and none of it should be added.

