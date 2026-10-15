# Invariants

- This directory tests the Rust path and nothing else. Treat the Python implementation of every route as already removed. Never call it on purpose, never run one request through both engines, and never compare Python and Rust outputs, headers, cache keys or logs
- A test that would still pass with Python serving the call does not belong here. Every test asserts the call was served natively with `support.routing.assert_served_natively`
- Select the Rust path by injecting catalog rules for the route through `support.routing.native_route(...)`. Never select a backend through environment rollout flags, public rollout APIs or mutation of catalog rule globals. The rollout in `litellm/rust_bridge/catalog.py` is a product decision, and a test here must pass no matter what it says, including for a route marked `PYTHON_ONLY`
- A test whose subject is the rollout switch itself (environment, public rollout APIs, catalog rules) is a unit test of `litellm/rust_bridge/configuration.py` or `catalog.py` and lives in `tests/unit/rust_bridge/`
- A test lives here only when it cannot be written in Rust. That means it needs the Python public API, Python-owned state such as `litellm.cache` and callbacks, or the process boundary (fork, threads, event loops). Anything checkable from the Rust side goes in the crate's own tests
- The wire is the local recording server. No test here reaches a real provider
- Never parametrize one test across routes. Each route folder owns its tests, its `server` fixture, its request data and its `call`
- Call `_native` directly only for a case the public API cannot reach

# Layout

Each route folder (`ocr/`, `messages/`, `responses/`, `chat_completions/`) uses the same file names, one concern per file

- `test_dispatch.py` the call reaches the native route
- `test_requests.py` Python inputs that reach the wire
- `test_failures.py` errors raised to the caller and rejection before provider IO
- `test_settings.py` globals, env, named credentials and their precedence
- `test_streaming.py` stream relay, early close and stream-time logging
- `test_callbacks.py` the logger hook contract
- `test_body_hooks.py` handler body hooks such as interceptors and agentic loops
- `test_lifecycle.py` cancellation, concurrency, context and nested calls
- `test_cache.py` the configured cache seen from this route

Route data and the route's `call` live in `<route>/data.py`. `support/` holds only route-neutral helpers
