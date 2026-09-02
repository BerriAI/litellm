# Existing e2e SDK tests

Reruns the real provider e2e suites that already live under `tests/` — for
example `tests/ocr_tests/`, which calls `litellm.ocr()`/`litellm.aocr()`
against live Mistral, Azure AI, Azure Document Intelligence, and Vertex AI
endpoints — once with the Rust bridge disabled and once with it enabled, via
each function's environment-variable toggle (`LITELLM_USE_RUST_OCR` for
OCR). Unlike `e2e_fuzz_tests/`, which fuzzes bridge-level inputs, this
strategy proves that pre-existing, human-written SDK-level tests keep
passing unmodified on both code paths. It requires the same provider
credentials the underlying suites already require and makes real,
billable API calls.
