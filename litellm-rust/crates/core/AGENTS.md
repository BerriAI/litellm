litellm-core is the PURE translation layer — types, route contracts (traits), provider transforms (modules under `providers/`), and the router. No network, no I/O, no env reads.

Routes (ocr, realtime) and providers (mistral, openai) are modules, not crates.
