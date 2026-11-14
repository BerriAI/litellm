# CustomLoggerV2

The V2 callback API. A callback declares which hooks it subscribes to, which fields it reads and writes, and whether it may abort. Registration turns that into a frozen manifest, and the dispatcher plans and runs every callback in `litellm.callbacks` in list order from those declarations.

## How the three kinds of callback fit

- **V2 callbacks** subclass `CustomLoggerV2` and declare handlers with `@on`.
- **Legacy `CustomLogger`s** are wrapped one per instance by `legacy.py`. Each declares a write to `deprecated_kwargs`, so it gets its own stage, and legacy and V2 callbacks share one ordered line per hook.
- **Rust callbacks** implement `trait Callback` in `litellm/rust/crates/callbacks`. `rust.py` builds a `CustomLoggerV2` from the Rust manifest, so a Rust callback runs at its position in the list like any other. `contract.json` and `records.json` in that crate pin the hooks, fields and records, and tests on both sides rebuild them.

Typed fields that are not plain request keys are worked out from the hook's arguments. `derivations.py` holds those derivations, and `field_sources.py` lays a table of them over a hook's fields. `litellm/proxy/v2_proxy_fields.py` holds the ones only the proxy can derive.

## Modules

| Module | What it holds |
|---|---|
| `enums.py` | `Hook`, `Field` (with its bit and cost tier), `Activation`, `OptOutBy` |
| `hooks.py` | `HOOK_SPECS`: per hook, the fields it may write, the outcomes it allows, and whether it is on the request path |
| `logger.py` | `CustomLoggerV2` and the `@on` decorator |
| `manifest.py` | The frozen per-instance manifest built at registration |
| `outcomes.py` | `Patch`, `Abort`, `Respond`, `Reroute` |
| `patches.py` | `LiveFields`, the field views handlers read, and applying a `Patch` to them |
| `planner.py` | Ordering a hook's subscribers into blocking stages, awaited readers and off-path deliveries |
| `dispatcher.py` | Running a plan for one request: `Dispatcher`, `RequestScope`, `HookResult` |
| `registry.py` | The registered callbacks and per-hook snapshots |
| `selection.py` | Which callbacks a request runs: activation, no-log and caller opt-outs |
| `host.py` | The process-wide dispatcher and the request scope bound to the current context |
| `offpath.py` | The sink that delivers off-path subscribers after the hook returns |
| `enforcement.py` | `ContractViolation` and `UndeclaredReadError` |
| `call.py` | `CallFacts`, the ids, kind, target and timing every handler receives |
| `legacy.py` | Wrapping legacy `CustomLogger`s and building and running a hook's line at each call site |
| `hook_point_fields.py` | Field views for the router and deployment hook points |
| `field_sources.py` | `FieldSource`: how one field is derived from a hook's arguments, or written back |
| `derivations.py` | The derivations and the per-hook tables of them |
| `rust.py` | `rust_logger`: a Rust callback as a `CustomLoggerV2`, and abort error factories |

## Where it is wired in

- Proxy hooks: `litellm/proxy/utils.py`, and `bind_v2_request_scope` in `litellm/proxy/litellm_pre_call_utils.py`
- Logging plane: `litellm/litellm_core_utils/litellm_logging.py`
- SDK deployment hooks: `litellm/utils.py`, `llm_http_handler.py` and the Anthropic native handler
- Router: `filter_deployments` and the fallback events
