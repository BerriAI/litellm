The shared callback contract lives in `callback_contract.py`. Each route's `test_callbacks.py` imports `TestCallbackContract` and defines a `callback_route` fixture returning a `CallbackRoute`. Import the class directly, without subclassing it, so every new shared case is collected for every route

Adapters in `callback_routes.py` supply public entrypoints, request and response fixtures, call types, and response text/replacement operations. The invocation forces `RUST_REQUIRED`, so a Python fallback cannot satisfy a native callback test. Expectations concern behavior owned by LiteLLM, not vendor pricing or defaults

The adapters cover the four inference routes using the shared native legacy-callback lifecycle: OCR, Messages, Chat Completions, and Responses. Embeddings still declines native execution. Transcription uses an older direct bridge and is not covered by this contract yet

The shared cases exercise sync and async calls, global and request callback registration, terminal success and failure, exception identity, duplicate registration, request/header edits, observer failures, deployment rejection and replacement, caller context, cancellation, and deferred release/discard. Assertions run against real callback objects and a local HTTP server, not mocked logging methods

OCR previously had broader callback and lifecycle coverage: aliasing, mutation, cancellation, response replacement, deferred release, and observer failures. Messages mainly exercised request edits, ordinary success/failure, and streaming. Neither suite compared these hooks against Python, and OCR's deferred-release test could not detect that the proxy released only OCR calls. Sharing the cases exposed that restriction on three other routes, missing logger access in deployment hooks on all four routes, and deferred logging lost after a Messages stream closed early

Streaming routes also import `TestStreamCallbackContract` and provide `callback_stream_response`. Those cases cover full consumption, early close, close before consumption, transport failure, deferred logging, repeated release, and repeated close. Only Messages currently exposes streaming through the native Python host

Messages-specific tests run the same active callback against the Python implementation and the native route. Three native cases are strict expected failures: request hooks are omitted, agentic completion hooks are omitted, and pre-API logging has a different input/credential contract. Unexpected exceptions still fail the tests. A fix producing XPASS fails the suite until its expected-failure marker is removed

Run the four route suites against a compiled extension with:

```sh
LITELLM_RUST=1 LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest \
  tests/test_litellm_rust/ocr/test_callbacks.py \
  tests/test_litellm_rust/messages/test_callbacks.py \
  tests/test_litellm_rust/chat_completions/test_callbacks.py \
  tests/test_litellm_rust/responses/test_callbacks.py
```

This is a shared lifecycle contract, not a claim of complete route parity. Provider transformations, interceptors, cache-control injection, and route-specific callback payload fields need their own behavioral cases. Keep those alongside the route's shared-suite import
