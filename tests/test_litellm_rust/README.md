# Rust OCR bridge tests

This suite covers OCR requests through LiteLLM's compiled Rust extension. OCR behavior tests live under `ocr/`; reusable OCR request, callback, and recording-server fixtures live under `support/`

A test name identifies the OCR entrypoint or callback under test and its expected observable result. Parameter IDs state the execution mode or credential case. Keep multiple assertions together only when they prove one request, mutation, failure, or callback lifecycle behavior. Record callback observations and assert them after the callback returns because production logging can swallow callback exceptions

`ocr/test_requests.py` covers provider payloads, file preparation, endpoint and credential resolution, normalized responses, errors, timeouts, and Azure token-provider behavior. `ocr/test_callbacks.py` covers OCR callback inputs, mutations, ordering, context, failure handling, concurrency, and cleanup. `ocr/test_guardrails.py` covers OCR post-call blocking and response replacement. `ocr/test_dispatch.py` covers public sync and async native dispatch and explicit Python dispatch. `test_ocr.py` is the strict wire-level smoke test

Run `make test-rust-extension` as the acceptance command. It builds a fresh wheel, installs that wheel into a temporary environment, requires `LITELLM_RUST=1`, and runs this suite with isolated Python imports

Collection fails when `LITELLM_RUST=1` is set but the compiled `_native` module cannot be imported. The autouse fixture selects Rust for every test unless a fallback or parity case explicitly selects Python. Backend selection alone does not prove native execution because a public OCR request can fall back. Tests that claim native dispatch assert either the Rust response marker or a wire property that distinguishes the native client. `test_public_ocr_executes_the_compiled_extension_without_python_fallback` is strict and its server rejects Python HTTPX requests

The OCR contract modules are non-strict expected failures until the retained callback implementation from #40070 lands. The compiled-extension OCR smoke test remains strict. Passing contract cases appear as XPASS so staging coverage stays visible
