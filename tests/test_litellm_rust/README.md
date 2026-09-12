# Rust OCR bridge tests

This suite covers OCR requests through LiteLLM's compiled Rust extension. OCR behavior tests live under `ocr/`; reusable OCR request, callback, and recording-server fixtures live under `support/`

A test name identifies the OCR entrypoint or callback under test and its expected observable result. Parameter IDs state the execution mode or credential case. Keep multiple assertions together only when they prove one request, mutation, failure, or callback lifecycle behavior. Record callback observations and assert them after the callback returns because production logging can swallow callback exceptions

`ocr/test_requests.py` covers provider payloads, file preparation, endpoint and credential resolution, normalized responses, errors, timeouts, and Azure token-provider behavior. `ocr/test_callbacks.py` covers callback inputs, mutations, ordering, context, failure handling, concurrency, and cleanup. `ocr/test_guardrails.py` covers post-call blocking and response replacement. `ocr/test_lifecycle.py` checks final object identity, finalization failures, caller-task context, cancellation, nested requests, executor scheduling, and deferred release. These tests use the public OCR APIs. `ocr/test_dispatch.py` covers enabled native dispatch and disabled legacy dispatch. `test_ocr.py` exercises the compiled Rust transport directly

Run `make test-rust-extension` as the acceptance command. It builds a fresh wheel, installs that wheel into a temporary environment, requires `LITELLM_RUST=1`, and runs this suite with isolated Python imports

Collection fails when `LITELLM_RUST=1` is set but the compiled `_native` module cannot be imported. The autouse fixture isolates callbacks, both logging executor references, and configuration state. All tests are strict

The native lifecycle accepts the existing Mistral, Azure, and Vertex admission surface. Calls using Python-only Azure credential providers or caching select legacy execution before admission. Azure token-provider tests cover that compatibility fallback. Rust lifecycle ordering lives in `core/src/call_lifecycle/host.rs`; shared Python logging operations live in `python-bridge/src/lifecycle.rs`; the caller-task coroutine protocol lives in `python-interop/src/coroutine.rs`
