# SDK function tracing

`profile_python` records selected Python function calls by code object, including their order and nesting depth. `assert_function_trace_parity` compares those events with the Rust trace supplied by a scenario. The profiler observes the current thread; functions dispatched to executor threads need profiling in those threads. Depth counts only selected ancestors. The shared mock provider requires exactly one HTTP request per invocation

The Rust bridge's `function_trace` module records TRACE spans targeting `litellm::function_trace`. Use `#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]` to take names from the actual functions without recording arguments. Attach its dispatcher to each future with `WithSubscriber` so concurrent requests keep separate traces

This layer provides reusable tracing infrastructure. It does not enable tracing on routes, add native Python exports, validate function signatures, or claim provider parity. Route-specific instrumentation and parity scenarios are added by later layers

Run the infrastructure tests with `python -m pytest tests/sdk_function_trace/test_profiler.py tests/sdk_function_trace/test_mock_provider.py` and `cargo test -p litellm-python-bridge function_trace` from `litellm-rust`

To list the Python functions for a route without installing LiteLLM, run this from the repository root:

```bash
python3 tests/sdk_function_trace/list_python_steps.py --route chat_completions --signatures --calls
```

Routes are `chat_completions`, `audio_transcription`, `messages`, and `ocr`; omit `--route` to list all. Omit `--signatures` or `--calls` for shorter output. `--repo /path/to/litellm` reads another checkout

The listing reads source declarations and direct call sites, including conditional branches. It covers SDK entrypoints, shared handlers and bases, and selected provider paths. It reports missing selected functions as `NOT FOUND`; it does not execute calls or claim runtime ordering
