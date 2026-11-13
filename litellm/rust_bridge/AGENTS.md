# Ownership

- This package owns
  - choosing Rust or Python for each call, and falling back to Python when Rust is unavailable
  - Python public API compatibility at the native boundary
  - projecting Python settings into the values Rust reads
  - the Python bindings handed to the Rust bridge
- It does not own
  - provider execution, which is Rust core
  - CPython runtime mechanics, which is `litellm-host-python`
  - legacy callback sharing and dispatch policy, which is `callbacks-legacy-python`
- `host/` holds the modules Rust imports by name to reach Python-owned state and behavior. Only Rust calls them, and Rust reaches Python nowhere else. `host/AGENTS.md` states that contract

# Routing

- `catalog.py` decides, the detectors describe the request, `dispatch.py` applies the decision
- Shared detectors live in `detectors.py`; route-only detectors live in `<route>/detectors.py`
- The first matching rule wins. A context no rule matches stays on Python
- Every Python-only exclusion for a route (a `targets` or `features` rule) sits above any Rust rule for that route
- A route whose Python side is `NO_PYTHON` resolves to `RUST_REQUIRED` for every context it can produce
  - `tests/test_litellm_rust/test_public_entries.py` marks such a route `python=None`, so dropping a route's Python changes its entries, its rules and that field together
- A detector reads the same inputs the legacy Python path reads, so the catalog and the fallback agree on which requests need Python
  - A route's context, its target as well as its features, reads `NativeCall.resolved`, so an argument counts the same whether it came by position, by keyword or as a signature default
  - A route's sync and async dispatches read different detector maps when the legacy handlers differ, as the sync Responses handler runs the deployment hook itself. Messages reads one map for both, since Rust serves its async-only pre-request hooks, web search short-circuit and interceptors through host calls
- A detector that raises keeps the call on Python. `PublicDispatch` logs it and never fails the call over routing
- Messages for provider `anthropic` is `RUST_REQUIRED` and has no Python path
  - a call the rules would send to Python (a `MESSAGES_PYTHON_FEATURES` feature or a raising detector) raises `UnsupportedParamsError` from the Python fallback in `litellm/messages/dispatch.py`, which names the detected features
  - other Messages providers keep the Python fallback
- `catalog.RULES` is rollout policy with no tests of its own. Tests pass the rules they need, so check these invariants by review whenever `RULES` changes

# Public entries

- Every route serves `a<entry>(...)`, `<entry>(..., a<entry>=True)` and `a<entry>(..., a<entry>=True)` the same way, handing native one identical `NativeCall` with no async flag on the async binding
- `tests/test_litellm_rust/test_public_entries.py` holds this for every route in its table, so a new public route gets a row there

# Execution

- `execution.py` owns generic inline execution, stream iteration and the step classes Rust constructs
- Native entrypoints return an unstarted `Execution`. `execution.synchronous` or `execution.asynchronous` hands it to the caller through `run` or `drive`, so Rust never names a Python driver or stream class
- Keep one driver, implementing generator-style `send` (`send(None)` starts), `throw` and idempotent `close`
- Generic steps and suspension handling never depend on LiteLLM response metadata. Public stream construction, `_hidden_params` and header compatibility stay in the product wrappers, and generic stream heads stay opaque
- Await each selected suspension in the caller's task, so hooks keep their thread, loop and context
- Return a final awaitable as a value instead of awaiting it implicitly
- Closing, cancellation and `GeneratorExit` release the execution without replaying provider work or starting more callbacks

# Fallback

- Native admission may pick Python before execution starts
- A failure after execution starts is terminal and never triggers fallback
- Creating an unstarted native coroutine acquires no clients or credentials and captures no execution context

# The `LITELLM_RUST` switch

- `decision` treats a blank or invalid `LITELLM_RUST` as unset and caches the parse by raw value. It runs inside SDK calls, so it never fails one over a switch and never validates per call
- It does not put the Rust gateway in front of Python. That is `LITELLM_RUST_FIRST`, which `litellm/proxy/rust_forward/mode.py` reads strictly at startup with this same `RustSwitch` type, stopping the CLI with one line naming the variable

# Native extension

- `_native` ships in every wheel and the sdist builds it, so it is always importable
- Import it with `from litellm.rust_bridge import _native` and use the attributes `_native.pyi` declares
- Never add a `None` branch, `getattr` default or stand-in class for a missing extension

# Verification

- Check cross-language behavior against a freshly installed extension, covering unstarted-call release, setup ordering, exception identity, sync and async streams, cancellation and GC
- Keep `_native.pyi` accurate about which APIs return a Future and which return a coroutine
- Unit tests of this package live in `tests/unit/rust_bridge/`, end-to-end Rust path tests in `tests/test_litellm_rust/`
