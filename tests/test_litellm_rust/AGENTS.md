# Scope

- This directory tests the Rust path end to end through the Python public API, plus the dispatch tests below, and nothing else
- A test belongs here only when it cannot be written in Rust, meaning it needs
  - the Python public API
  - Python-owned state such as `litellm.cache` and callbacks
  - the process boundary (fork, threads, event loops)
- Anything checkable from the Rust side goes in the crate's own tests
- The wire is the local recording server. No test here reaches a real provider

# Rust only

- Outside the dispatch tests, treat every route's Python implementation as already removed
  - never call it on purpose
  - never run one request through both engines
  - never compare Python and Rust outputs, headers, cache keys or logs
- A test that would still pass with Python serving the call does not belong here
- Every test except the dispatch tests asserts native serving with `support.routing.assert_served_natively`

# Dispatch tests

- They check which path a public call takes under given rules
  - each route folder's `test_dispatch.py`
  - `test_dispatch.py` at the top, which tests `PublicDispatch` with stub routes
  - `test_public_entries.py` at the top, which runs every route's public call forms through one table
- They pass `rules=` to a dispatch directly or wrap the whole test in one `support.routing.routed(...)` block, and they stub the native bindings
- They may call Python, since the choice between the two paths is what they test
- They live here until they can run in `tests/unit/` without the compiled extension

# Selecting a path

- Use `support.routing.native_route(...)`, which prepends `RUST_REQUIRED` rules to `catalog.RULES` and restores them on exit
  - route folders apply it to every test through the autouse fixture in their `conftest.py`
  - a test outside a route folder wraps its whole body in one `native_route(...)` block
  - a child interpreter script assigns `catalog.RULES` once at its top
- Dispatch tests use `support.routing.routed(...)` instead, which sets `catalog.RULES` to exactly the rules given
- Open either once per test, never per call. They swap a module global, so overlapping blocks (one per task in an `asyncio.gather`) restore out of order and leak rules into later tests
- Never select a backend through `LITELLM_RUST`, `litellm.rust(...)` or any other rebinding of `catalog.RULES`
- No test here depends on the rollout switch
  - only no rule, `PYTHON_ONLY` and `RUST_REQUIRED` pick a path without it, so those are the only rollouts used here
  - Rust with Python fallback needs the switch, so it is not reachable from this directory
  - what `LITELLM_RUST` and `litellm.rust(...)` mean is tested in `tests/unit/rust_bridge/test_configuration.py`, mostly through `decide`, which takes both as plain values
- Tests pass whatever the shipped rollout in `litellm/rust_bridge/catalog.py` says, including for a `PYTHON_ONLY` route

# Layout

- One folder per route (`ocr/`, `messages/`, `responses/`, `chat_completions/`), all using the same file names
  - `test_dispatch.py` which path the rules pick for each public call, and that the call reaches the native route
  - `test_requests.py` Python inputs that reach the wire
  - `test_failures.py` errors raised to the caller and rejection before provider IO
  - `test_settings.py` globals, env, named credentials and their precedence
  - `test_streaming.py` stream relay, early close and stream-time logging
  - `test_callbacks.py` the logger hook contract
  - `test_body_hooks.py` handler body hooks such as interceptors and agentic loops
  - `test_lifecycle.py` cancellation, concurrency, context and nested calls
  - `test_cache.py` the configured cache seen from this route
- `embeddings/` holds only `test_dispatch.py` so far
- `<route>/data.py` holds the route's request data and its `call`
- `<route>/conftest.py` owns the `server` fixture and the autouse `native` fixture
- `support/` holds only route-neutral helpers
- Never parametrize one test across routes, except in `test_public_entries.py`, whose subject is that every route serves its public call forms the same way
- Call `_native` directly only for a case the public API cannot reach
