# SDK function tracing

The compare runner executes the same SDK calls through the Python engine and the Rust native bridge against a local HTTP provider fixture, then prints each engine's pipeline step tree (steps missing on one side are marked and colored blue for python-only, yellow for rust-only) and a difference summary (shared step order, python-only steps, rust-only steps). Each invocation must issue exactly one HTTP request. It requires the LiteLLM Python dependencies and the native extension built with tracing support

From the repository root, using the project's Python environment:

```bash
uv run tests/sdk_function_trace/compare.py
uv run tests/sdk_function_trace/compare.py --route ocr
uv run tests/sdk_function_trace/compare.py --route ocr --sync
uv run tests/sdk_function_trace/compare.py --route all --both --check
```

Calls default to async; use `--sync` for synchronous calls or `--both` for the complete matrix. Python sync Messages raises `not implemented for sync calls`; only that exact failure is marked `SKIP`, and the runner still executes Rust sync Messages and subsequent routes. Bedrock transcription has no independent Python provider implementation: its Python trace covers SDK dispatch into Rust

Both engines are projected onto a shared per-route step table (`steps.py`): canonical names such as `transform_ocr_request` map Python functions (`MistralOCRConfig.transform_ocr_request`) and Rust spans (`transform_ocr_request`) to the same label. Only the first occurrence of each step is kept. Python indentation uses each event's actual frame ancestors and the nearest already displayed ancestor, so returned helpers and coroutine resumptions do not create false parents. Rust indentation uses instrumented span ancestry. Unmatched Rust span names pass through unchanged. `--full` prints every captured runtime event; validation still uses projected steps

Every report checks required stage presence and dependency order. Provider lookup must precede request transformation, which must precede HTTP, followed by response transformation. The handler must precede HTTP; parameter mapping and supported-parameter checks must precede request transformation. Environment validation and URL construction, where mapped, must precede HTTP. Python transcription is checked only through native dispatch. `--check` also requires identical canonical step sequences for comparable routes and exits nonzero for missing, extra, or reordered steps, or an unexpected call failure, after finishing all selected cases

Individual stage checks are separate from cross-language `step parity`. Passing stage checks cannot override a failing step comparison. Bedrock transcription and Python sync Messages report `UNAVAILABLE` for cross-language parity because they lack an independent Python execution to compare. Absolute nesting depth is not a cross-language gate: async Python Messages dispatches its handler onto another thread. See `route-comparison.md` for the audited matrix and remaining contract limitations

The Python runner uses the existing `profile_python` / `sys.setprofile` collector, selecting executed code under the installed `litellm` source directory instead of maintaining a function-name allowlist. It prints source locations and qualified function names, including repeated calls. Coroutine resumptions are counted once per invocation. It profiles the current thread and threads created during the call, including the fresh async executor. Existing worker threads are not retroactively profiled; background Python calls may appear, and indentation follows selected Python stack ancestors within each thread

The Rust runner calls the compiled PyO3 SDK entrypoints with `trace=True`. The existing `FunctionTrace` subscriber collects `#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]` spans for the route entrypoint, preparation, provider lookup, HTTP handler, and selected provider transformations. The shared `http_request` helper instruments the existing Rust send operation without changing clients, timeouts, signing, or error mapping. Function names come from the actual functions. `WithSubscriber` attaches the collector to each future across async polls. Arguments and provider payloads are not recorded in trace events. Uninstrumented functions do not appear; this is scoped instrumentation, not an exhaustive native call graph

Tracing is opt-in: native calls without `trace=True` keep their original response shape. Traced calls return `{"response": ..., "trace": [{"function": ..., "depth": ...}]}`. The runners print only trace events. Missing native support or empty traces fail instead of falling back to source searching. The old `--repo`, `--signatures`, and `--calls` options are removed

`profile_python(functions)` still supports direct function references for focused parity checks. `assert_function_trace_parity` compares selected Python events with Rust events supplied by an executable scenario. Successful stage checks prove the declared pipeline ran in a valid dependency order for this fixture; they do not assert identical function contracts, request bodies, responses, streaming behavior, or live-provider correctness

Build the extension with `maturin develop` in the project's virtual environment. Then run either command above to get the executed function order
