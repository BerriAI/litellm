# SDK function tracing

The compare runner executes the same SDK calls through the Python engine and the Rust native bridge against a local HTTP provider fixture, then prints both pipeline step trees and a difference summary (shared step order, python-only steps, rust-only steps). Each invocation must issue exactly one HTTP request. It requires the LiteLLM Python dependencies and the native extension built with tracing support

From the repository root, using the project's Python environment:

```bash
uv run tests/sdk_function_trace/compare.py
uv run tests/sdk_function_trace/compare.py --route ocr
uv run tests/sdk_function_trace/compare.py --route ocr --sync
```

Calls default to async; use `--sync` for synchronous calls. The Python Messages SDK currently raises `not implemented for sync calls`; the runner propagates that failure when explicitly requested. Rust Messages supports both modes

Both engines are projected onto a shared per-route step table (`steps.py`): canonical names such as `transform_ocr_request` map Python functions (`MistralOCRConfig.transform_ocr_request`) and Rust spans (`transform_ocr_request`) to the same label, so the two trees and the diff are directly comparable. Only the first occurrence of each step is kept, and indentation is rebuilt from the kept ancestors, so repeated logging and cost-calculation calls do not pollute the report. Python functions with no Rust counterpart appear as python-only steps (`get_provider_ocr_config`, `validate_environment`, `complete_url`, `http_request`); Rust spans with no Python matcher appear as rust-only steps (`prepare_chat_completions_call`). Unmatched Rust span names pass through unchanged so new instrumentation stays visible. `--full` prints every captured Python runtime event instead of the projected steps

The Python runner uses the existing `profile_python` / `sys.setprofile` collector, selecting executed code under the installed `litellm` source directory instead of maintaining a function-name allowlist. It prints source locations and qualified function names, including repeated calls. Coroutine resumptions are counted once per invocation. It profiles the current thread and threads created during the call, including the fresh async executor. Existing worker threads are not retroactively profiled; background Python calls may appear, and indentation follows selected Python stack ancestors within each thread

The Rust runner calls the compiled PyO3 SDK entrypoints with `trace=True`. The existing `FunctionTrace` subscriber collects `#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]` spans for the route entrypoint, preparation, HTTP handler, and selected provider transformations. Function names come from the actual functions. `WithSubscriber` attaches the collector to each future across async polls. Arguments and provider payloads are not recorded in trace events. Uninstrumented functions do not appear; this is scoped instrumentation, not an exhaustive native call graph

Tracing is opt-in: native calls without `trace=True` keep their original response shape. Traced calls return `{"response": ..., "trace": [{"function": ..., "depth": ...}]}`. The runners print only trace events. Missing native support or empty traces fail instead of falling back to source searching. The old `--repo`, `--signatures`, and `--calls` options are removed

`profile_python(functions)` still supports direct function references for focused parity checks. `assert_function_trace_parity` compares selected Python events with Rust events supplied by an executable scenario. A successful listing only proves which instrumented functions ran for that fixture; it does not assert cross-language function or response parity

Build the extension with `maturin develop` in the project's virtual environment. Then run either command above to get the executed function order
