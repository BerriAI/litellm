# SDK function tracing

Both runners execute SDK calls against a local HTTP provider fixture and print functions observed during execution, in event order with nesting indentation. Each invocation must issue exactly one HTTP request. They require the LiteLLM Python dependencies; Rust tracing also requires rebuilding the native extension after changing instrumentation

From the repository root, using the project's Python environment:

```bash
python -m tests.sdk_function_trace.list_python_steps --route ocr
python -m tests.sdk_function_trace.list_rust_steps --route ocr
python -m tests.sdk_function_trace.list_python_steps --route ocr --sync
python -m tests.sdk_function_trace.list_rust_steps --route ocr --sync
```

Calls default to async; use `--sync` for synchronous calls. The Python Messages SDK currently raises `not implemented for sync calls`; the runner propagates that failure when explicitly requested. Rust Messages supports both modes

Direct script execution also works. Omit `--route` to run all four routes: `chat_completions` and `messages` use Anthropic, `ocr` uses Mistral, and `audio_transcription` uses Bedrock. Bedrock transcription is Rust-only in this checkout, so the Python trace covers its SDK dispatch and requires the native extension too. The fixtures use dummy credentials and loopback HTTP, with no paid provider requests

The Python runner uses the existing `profile_python` / `sys.setprofile` collector, selecting executed code under the installed `litellm` source directory instead of maintaining a function-name allowlist. It prints source locations and qualified function names, including repeated calls. Coroutine resumptions are counted once per invocation. It profiles the current thread and threads created during the call, including the fresh async executor. Existing worker threads are not retroactively profiled; background Python calls may appear, and indentation follows selected Python stack ancestors within each thread

The Rust runner calls the compiled PyO3 SDK entrypoints with `trace=True`. The existing `FunctionTrace` subscriber collects `#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]` spans for the route entrypoint, preparation, HTTP handler, and selected provider transformations. Function names come from the actual functions. `WithSubscriber` attaches the collector to each future across async polls. Arguments and provider payloads are not recorded in trace events. Uninstrumented functions do not appear; this is scoped instrumentation, not an exhaustive native call graph

Tracing is opt-in: native calls without `trace=True` keep their original response shape. Traced calls return `{"response": ..., "trace": [{"function": ..., "depth": ...}]}`. The runners print only trace events. Missing native support or empty traces fail instead of falling back to source searching. The old `--repo`, `--signatures`, and `--calls` options are removed

`profile_python(functions)` still supports direct function references for focused parity checks. `assert_function_trace_parity` compares selected Python events with Rust events supplied by an executable scenario. A successful listing only proves which instrumented functions ran for that fixture; it does not assert cross-language function or response parity

Build the extension with `maturin develop` in the project's virtual environment. Then run either command above to get the executed function order
