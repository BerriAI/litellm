# Python boundary

This package owns native rollout and fallback selection, Python public API compatibility, settings projection and the Python bindings supplied to the Rust bridge. Rust core owns provider execution. `litellm-host-python` owns CPython runtime mechanics. `callbacks-legacy-python` owns legacy callback sharing and dispatch policy

`host/` holds the Python modules Rust imports by name to read Python-owned state (provider resolution, model capabilities, settings, caches, diagnostics, legacy callbacks, the secret manager, transport and timeout conversions). Its AGENTS.md states that contract

`lifecycle.py` owns generic inline execution and stream iteration, plus the step classes Rust constructs. Native entrypoints return an unstarted `Execution`, and `public_execution.py` hands it to the public caller through `lifecycle.run` or `lifecycle.drive` with the product stream wrappers, so Rust never names a Python driver or stream class. Keep one driver implementing generator-style `send` (`send(None)` starts), `throw` and idempotent `close`. Do not create a second implementation

Generic execution steps and inline suspension handling must not depend on LiteLLM response metadata. Public stream construction, `_hidden_params` and header compatibility remain product responsibilities. Keep generic stream heads opaque and preserve public header behavior in the product wrappers

Await each selected suspension in the caller's task so hooks retain thread, loop and context identity. A final awaitable value is returned as a value, not awaited implicitly. Closing, cancellation and `GeneratorExit` release the execution without replaying provider work or starting further callbacks

Native admission may select legacy before execution starts. A failure after execution starts is terminal and never authorizes fallback. Creating an unstarted native coroutine must not acquire clients or credentials or capture execution context

Verify cross-language behavior using a fresh installed extension, including unstarted-call release, setup ordering, exception identity, sync and async streams, cancellation and GC. Keep `_native.pyi` accurate about Future-returning and coroutine-returning APIs
