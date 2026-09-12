# Rust OCR bridge tests

This suite covers OCR requests through LiteLLM's compiled Rust extension. OCR behavior tests live under `ocr/`; reusable OCR request, callback, and recording-server fixtures live under `support/`

A test name identifies the OCR entrypoint or callback under test and its expected observable result. Parameter IDs state the execution mode or credential case. Keep multiple assertions together only when they prove one request, mutation, failure, or callback lifecycle behavior. Record callback observations and assert them after the callback returns because production logging can swallow callback exceptions

`ocr/test_requests.py` covers provider payloads, file preparation, endpoint and credential resolution, normalized responses, errors, timeouts, and Azure token-provider behavior. `ocr/test_callbacks.py` covers callback inputs, mutations, ordering, context, failure handling, concurrency, and cleanup. `ocr/test_guardrails.py` covers post-call blocking and response replacement. `ocr/test_lifecycle.py` checks final object identity, finalization failures, caller-task context, cancellation, nested requests, executor scheduling, deferred release, Reducto upload/parse and Azure submission/poll boundaries. These tests use the public OCR APIs. `ocr/test_dispatch.py` covers enabled native dispatch and rejection when native execution is disabled. `test_ocr.py` exercises the compiled Rust transport directly

Run `make test-rust-extension` as the acceptance command. It builds a fresh wheel, installs that wheel into a temporary environment, requires `LITELLM_RUST=1`, and runs this suite with isolated Python imports

Collection fails when `LITELLM_RUST=1` is set but the compiled `_native` module cannot be imported. The autouse fixture isolates callbacks, both logging executor references, and configuration state. All tests are strict

The native lifecycle supports Mistral, Azure, Vertex and Reducto workflows. Caller-supplied synchronous Azure token providers execute inline when requested by core. Disabled or unavailable native execution and unsupported caching requests raise an error rather than invoking a legacy Python OCR provider

Rust lifecycle ordering lives in `core/src/call_lifecycle/host.rs`, with provider work owned by `core/src/ocr/lifecycle.rs`. The native execution handle and Python reference ownership live in `python-bridge/src/lifecycle.rs`. One ordinary Python coroutine in `litellm/rust_bridge/lifecycle.py` awaits Rust-selected operations in the caller task through `start`, `resume_value`, `resume_error` and idempotent `close`. Tagged Await/Complete steps preserve awaitable final values. The hand-written Rust coroutine protocol has been removed

The extension explicitly requires the GIL and detaches Rust-only synchronous waits. Native results stay in Rust, while retained Python roots and exceptions participate in GC. Request projection and file reads happen after lifecycle setup and applicable deployment hooks. Cancellation during failure logging propagates, while deployment-failure observers preserve the original provider error. Native cancellation waits for the owned provider task through core; synchronous close and GC signal cancellation without claiming to await termination

The native-backed driver probe is `litellm-rust/crates/python-bridge/tests/lifecycle.py`, invoked by Rust unit tests. It covers custom awaitables, task/thread/loop identity, context writes, exception identity, repeated cancellation, re-entry and cycles. Native typing, serialization benchmark additions and token-counter changes are separate follow-ups

## Local validation

The final lifecycle validation run passed `cargo fmt --check`, workspace Clippy with warnings denied, core Clippy with `bedrock-auth`, gateway Clippy with all features, workspace tests, core tests with `bedrock-auth`, and gateway tests with `server`. The installed-wheel acceptance command passed 109 tests on GIL-enabled CPython 3.12.13 with the ABI3 extension. Focused Ruff and basedpyright checks also passed

The installed-wheel run reported two existing Pydantic warnings that ReadOnly TypedDict fields are not runtime mutation guards. Credential-dependent live Bedrock and OpenAI realtime Rust tests remained explicitly ignored. These local results cover controlled provider dependencies and do not establish live-provider or free-threaded Python acceptance
