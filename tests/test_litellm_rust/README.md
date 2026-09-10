# Rust OCR bridge tests

This suite covers OCR requests through LiteLLM's compiled Rust extension. OCR behavior tests live under `ocr/`; reusable OCR request, callback, and recording-server fixtures live under `support/`

A test name identifies the OCR entrypoint or callback under test and its expected observable result. Parameter IDs state the execution mode or credential case. Keep multiple assertions together only when they prove one request, mutation, failure, or callback lifecycle behavior. Record callback observations and assert them after the callback returns because production logging can swallow callback exceptions

`ocr/test_requests.py` covers provider payloads, file preparation, endpoint and credential resolution, normalized responses, errors, timeouts, and Azure token-provider behavior. `ocr/test_callbacks.py` covers OCR callback inputs, mutations, ordering, context, failure handling, concurrency, and cleanup. `ocr/test_guardrails.py` covers OCR post-call blocking and response replacement. These contract modules call the Rust bridge directly. `ocr/test_dispatch.py` has the single public API dispatch test, covering enabled native dispatch and disabled Python dispatch. `test_ocr.py` is a strict smoke test of the compiled Rust OCR transport

Run `make test-rust-extension` as the acceptance command. It builds a fresh wheel, installs that wheel into a temporary environment, requires `LITELLM_RUST=1`, and runs this suite with isolated Python imports

Collection fails when `LITELLM_RUST=1` is set but the compiled `_native` module cannot be imported. The autouse fixture isolates callback and configuration state but does not select a backend. Native contract tests call `litellm.rust_bridge.ocr` directly, while the strict dispatch test explicitly enables and disables Rust and records which OCR entrypoint runs

The OCR contract modules are non-strict expected failures until the retained callback implementation from #40070 lands. The public dispatch test remains strict. Passing contract cases appear as XPASS so staging coverage stays visible
