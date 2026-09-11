# Rust OCR bridge tests

This suite covers OCR requests through LiteLLM's compiled Rust extension. OCR behavior tests live under `ocr/`; reusable OCR request, callback, and recording-server fixtures live under `support/`

A test name identifies the OCR entrypoint or callback under test and its expected observable result. Parameter IDs state the execution mode or credential case. Keep multiple assertions together only when they prove one request, mutation, failure, or callback lifecycle behavior. Record callback observations and assert them after the callback returns because production logging can swallow callback exceptions

`ocr/test_requests.py` covers provider payloads, file preparation, endpoint and credential resolution, normalized responses, errors, timeouts, and Azure token-provider behavior. `ocr/test_callbacks.py` covers OCR callback inputs, mutations, ordering, context, failure handling, concurrency, and cleanup. `ocr/test_guardrails.py` covers OCR post-call blocking and response replacement. These contract modules use the public SDK with native support required, plus direct `_native` calls for host-hook behavior. `ocr/test_dispatch.py` covers enabled native dispatch and disabled Python dispatch. `test_ocr.py` is a strict smoke test of the compiled Rust OCR transport

Run `make test-rust-extension` as the acceptance command. It builds a fresh wheel, installs that wheel into a temporary environment, requires `LITELLM_RUST=1`, and runs this suite with isolated Python imports

Collection fails when `LITELLM_RUST=1` is set but the compiled `_native` module cannot be imported. The autouse fixture isolates callback and configuration state but does not select a backend. Native request helpers enable Rust and reject unsupported configurations that would fall back to Python. The dispatch test records which OCR entrypoint runs

## Callback task and cancellation contract

Synchronous native calls invoke hooks on the caller thread in its current Python context. If called inside an asyncio task, callbacks see that same task and their context changes remain visible to the caller

Async native calls invoke synchronous hooks on the caller's event loop in separate tasks. Each token-provider or pre-call phase starts with a copy of the context captured at native entry. Context changes remain visible to later callbacks within the same pre-call phase, but do not propagate to another phase or back to the caller. Retained Python objects remain shared regardless of these context boundaries

Cancelling the native awaitable cancels pending callback delivery and eventually releases its references. A callback-raised `CancelledError` aborts the request before provider transport. Cancellation cannot interrupt a synchronous Python callback that is already executing; it must return or raise before its event loop can process cancellation

The pending-delivery tests use the event loop's task factory to hold the callback coroutine behind a gate. They wait for cancellation acknowledgement before checking cleanup, rather than relying on a fixed delay
