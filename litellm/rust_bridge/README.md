# Native route foundation

Each route has a Python package and a matching Rust module under `crates/python-bridge/src/routes/`. Python `__init__.py` files are thin entrypoints exporting `ROUTE: NativeRoute` and public adapters. Request/response protocols live in `types.py`, value adapters in `value.py`, callback adapters in `callbacks.py` where needed, and full-call bindings in `lifecycle.py`. Unimplemented routes add these files when they gain an implementation

WebSocket is a Responses transport: its adapters live in `responses/websocket.py` and Rust `routes/responses/websocket.rs`, using the Responses route policy. Token counting is a utility outside the route registry, in `token_counter.py` and Rust `src/token_counter.rs`

`configuration.py` owns release policy. OCR is default-on, Messages and other optional routes are default-off, and transcription is required-native. The process override takes precedence over the environment except for OCR's existing environment opt-out. Required-native execution ignores optional rollout switches. A default is an enablement choice, not a claim that a lifecycle implementation exists

`NativeRoute.select(binding)` checks policy before discovering the native module. `NativeBinding` handles validation and resettable overrides, with injectable discovery for tests. Native exports keep their existing names; moving a Python module into a package does not change its import path

## Lifecycle contract

Full-call bindings implement `NativeLifecycle[Request, Response]`: `(request, args, kwargs, asynchronous)`. The synchronous form returns a response, while the asynchronous form returns an inline-driven coroutine. Original positional arguments, keyword arguments and Python object identities stay available to the host

Core owns effect-free admission and callback sequencing. The PyO3 `PythonRoute` implementation retains Python objects, projects consumed fields and executes core-selected hooks. The shared native handle and Python `lifecycle.py` driver preserve caller task/context, error identity, cancellation and cleanup. Python logging continues to select registered integrations and their dispatch modes

Only disabled/unavailable native execution or a typed pre-effect admission decline permits Python fallback. Callback failures, projection errors and post-admission failures must not replay the request. Keep success/failure dispatch after fallible response finalization

OCR implements this contract today. Messages, chat completions and transcription retain their existing value-based execution while their new full-call lifecycle slots are unfinished. Embeddings, rerank, image generation/edit, speech, moderation and Responses have lifecycle slots but no public SDK wiring here. The Rust `unimplemented_lifecycle_route!` macro registers each slot and maps a pure core decline to `RustBridgeDeclined`, without inspecting the request. Deliberately avoid `todo!()` in Python-callable paths because it panics instead of providing safe admission fallback

## Extending a route

Replace the route's Rust lifecycle stub with a typed core call and a `PythonRoute` host, following OCR's `project`, `callbacks` and `lifecycle` split. Give its Python binding concrete request/response types, wire the public entrypoint through admission-only fallback, and prove positive native execution and callback parity before changing its release default

Streaming and WebSocket sessions do not yet use the full-call lifecycle contract. Their follow-up needs explicit chunk delivery, backpressure, final response aggregation, consumer close, cancellation acknowledgement, deferred terminal dispatch and exactly-once cleanup. Returning an iterator or opening a socket is not terminal success. WebSocket uses the Responses policy; token counting keeps the global optional switch. Both use shared loading and retain their own session/utility protocols
