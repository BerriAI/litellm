This directory holds only the tests that cannot be written in the Rust code

Each route folder (`ocr/`, `messages/`, `responses/`, `chat_completions/`) uses the same file names, one concern per file

- `test_dispatch.py` which path served the call
- `test_requests.py` Python inputs that reach the wire
- `test_failures.py` errors raised to the caller and rejection before provider IO
- `test_settings.py` globals, env, named credentials and their precedence
- `test_streaming.py` stream relay, early close and stream-time logging
- `test_callbacks.py` the logger hook contract
- `test_body_hooks.py` handler body hooks such as interceptors and agentic loops
- `test_lifecycle.py` cancellation, concurrency, context and nested calls

A concern that exists for only one route gets its own file there, never a provider-named one

- Route data and the route's `call` live in `<route>/data.py`, and `<route>/conftest.py` owns the `server` fixture
- `support/` holds only route-neutral helpers. Never parametrize one test across routes, except in feature folders such as `cache/`
- Call a public function through its route object in `litellm/<route>/dispatch.py` (`COMPLETION`, `ACOMPLETION`, `ANTHROPIC_MESSAGES`, ...) with `rules=support.routing.rust_required(...)` or `python_only(...)`. Never select a backend by rebinding the catalog's `RULES` or setting `LITELLM_RUST`. Call `_native` directly only for a case the route object cannot reach
- Check the native path with `support.routing.assert_served_natively`
