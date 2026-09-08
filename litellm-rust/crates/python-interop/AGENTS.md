litellm-python-interop is the domain-neutral PyO3 foundation.

- Owns generic Python/Serde conversion and interpreter primitives (`gil`, `marshal`)
- Depends on PyO3 but no LiteLLM domain crate; no route types, no API registration, no cdylib
- Keep it free of `litellm-core`, `OcrRequest`, `CallServices`, LiteLLM exceptions or `_native` surface
