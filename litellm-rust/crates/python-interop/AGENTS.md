litellm-python-interop is the domain-neutral PyO3 foundation.

- Owns generic Python/Serde conversion (`marshal`) and Python↔Tokio execution (`execution`)
- Depends on PyO3 but no LiteLLM domain crate; no route types, no API registration, no cdylib
- `execution` is generic over the caller's error type (`map_error: fn(E) -> PyErr`); the bridge passes its own mapper
- Keep it free of `litellm-core`, `OcrRequest`, `CallServices`, LiteLLM exceptions or `_native` surface
