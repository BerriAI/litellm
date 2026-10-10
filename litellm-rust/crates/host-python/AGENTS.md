- Target invariants. Implementation and runtime validation may lag these rules

## Boundary with Python consumers

This crate owns CPython execution mechanics for generic `litellm-host` machines and hooks. Consumers supply domain bindings, host operations, public result construction and callback policy. Neither Rust dependencies nor Python imports may require LiteLLM route modules or legacy `Logging`

`src/native.rs` belongs here: it runs a generic machine through the Python runtime and owns its pending execution and abort handle. Keep provider selection, request projection and public exception policy out of it. A rename to `machine_runner.rs` is optional and must not change behavior

The execution handle receives its Python lifecycle binding from its consumer through `PythonLifecycle` rather than import a fixed `litellm.rust_bridge` module. Generic suspension and execution state validation belong here. Public stream wrappers and `_hidden_params` conventions belong to the consumer

Creating a resolved asyncio Future from an already constructed Python value belongs here, alongside runtime waiting, interpreter detachment and panic containment. Choosing which callable exceptions become a public `RuntimeError` belongs to the consumer. `python-bridge::callable::wrap_failure` owns that policy

The driver owns ordering: start, argument preparation, prepared-argument hooks, binding decode and machine start. Fallible per-call resource setup supplied by the consumer runs after all argument hooks, using the prepared argument view, and before provider work. Setup failure follows the existing terminal failure path. Creating or discarding an unstarted coroutine must not initialize clients, acquire credentials or capture execution context

Boundary tests exercise behavior with a supplied lifecycle binding without importing the LiteLLM Python package. Pin inline awaiting, awaitable final values, exception identity, cancellation, re-entry and release of retained objects, rather than module names or source layout

`HookChain` composes Python runtime hooks in order. Each argument, wire-request and response transformation feeds its result to the next hook. After all argument transformations, the driver calls `arguments_prepared` on every hook in order. Retained callback views must adopt that dictionary before later policy hooks can mutate or reject it. SDK policy is supplied by bridge composition as a hook, never a separate driver phase or parameter. Hooks implement only the stages they need. Default stages preserve the supplied values

Terminal notifications share the selected response or exception. An ordinary notification error is reported as unraisable and does not skip the next hook or replace the selected outcome. Preparation, interception and transformation errors stop the chain. Cancellation stops all further hook dispatch. Suspensions stay inline in the existing driver, and the chain traverses retained event values for GC

## Existing runtime invariants

- Keep this crate the CPython runtime adapter and nothing more: Serde marshalling, interpreter detachment, tokio/asyncio glue, the `Execution` handle, the call driver and the `PythonBinding`, `PythonHostCalls` and `PythonOwned` traits, and the `PythonRuntime` specialization of `host::hooks::CallHooks`
  - No LiteLLM domain dependencies beyond `litellm-host`: no route types, no `Logging` policy, no public API registration, no cdylib build features
  - `PythonCallHooks` only constrains the shared call-stage interface to `PythonRuntime` and Python ownership. It must not redeclare the stages
  - `PythonCallEvent` is a specialization of the shared `CallEvent`, never a separately defined lifecycle. The driver emits `Succeeded` or `Failed` exactly once and never dispatches callbacks after a cancellation. Which Python objects consume those events is the legacy adapter's business
  - `CallOptions` can publish snapshots independently of callback delivery. Terminal snapshots follow completed hook dispatch, and cancellation never calls a Python callback. For Python-driven calls, leave the machine's observation publisher unset so the driver is the sole publisher of intercepted provider-response snapshots
  - `PythonBinding::decode_request` receives the keyword view returned by `prepare_arguments` and updated by `arguments_prepared`, not the caller's dict. A binding that decodes from it inherits all composed argument rewrites
  - A native failure, including one a host op returns as `InvokeError::Native`, is classified exactly once through the binding's `map_error`. A Python exception raised inside the call, and a failure in `prepare_arguments` or `transform_response`, is raised as is
  - A failing `map_error` is raised with the native error's text as its `__context__`, never swallowed
- Use standard PyO3 ownership and conversion APIs
  - Prefer `Bound<'py, T>` for attached operations/results, `Py<T>` for retention. Binding/unbinding does not copy payloads
  - Use `pythonize` for selected Serde data, never a JSON-text round trip. Share conversion with `Pythonized<T>`
  - Preserve `PythonizeError`'s standard conversion into `PyErr`. Do not stringify original Python exceptions into new `ValueError`s
  - Keep serializer-panic containment in `Pythonized<T>`: async output conversion can run in an unjoined blocking task and otherwise strand delivery
- Use `Python::detach` for Rust-only work. Python operations require attachment
  - Keep diagnostic counters in the consumer. Wrapper invocations do not measure every interpreter release
  - Release exclusive class borrows/locks before Python calls or decrements that can invoke finalizers. Expose retained Python edges to GC without calling Python during traversal
- Keep coroutine driving in the shared Python driver and the native handle
  - Shared driver implementation: `litellm/rust_bridge/lifecycle.py`, handle: `src/handle.rs`, call driver: `src/driver.rs`, native-backed behavior tests: `tests/lifecycle.py`. The consumer supplies the lifecycle binding
  - Every lifecycle suspension is awaited inline in the caller's task. `into_future` creates a separate task and cannot satisfy this contract
- References: [ownership](https://pyo3.rs/v0.29.2/types.html), [conversions](https://pyo3.rs/v0.29.2/conversions/traits.html), [pythonize errors](https://docs.rs/pythonize/0.29.0/src/pythonize/error.rs.html)
  - [GC](https://pyo3.rs/v0.29.2/class/protocols.html#garbage-collector-integration), [re-entry](https://pyo3.rs/v0.29.2/class/call.html), [parallelism](https://pyo3.rs/v0.29.2/parallelism.html), [async delivery source](https://docs.rs/pyo3-async-runtimes/0.29.0/src/pyo3_async_runtimes/generic.rs.html)
