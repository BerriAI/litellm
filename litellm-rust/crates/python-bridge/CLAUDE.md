# Python bridge

Follow `AGENTS.md` for the boundary invariants

## Ownership

Core owns typed native state, effect-free admission, lifecycle sequencing,
provider preparation and I/O, normalization, and dispatch decisions

This crate owns Python argument projection, retained Python references, public
response and exception construction, callback invocation, and host scheduling
Generic conversion and GIL utilities belong in `litellm-python-interop`

Only core admission may authorize legacy fallback. Execution and conversion
failures are terminal, including authentication and connection failures

## Structure

Each route registers its lifecycle binding from `routes/<route>/lifecycle.rs`
Unimplemented routes use `unimplemented_lifecycle_route!`. Keep value bindings
in `value.rs` where needed, and add projection or callback modules when the
route requires them. Register route functions through `definition::add_function`
to reject duplicate exports

`lifecycle/mod.rs` declares modules and exports the shared boundary types
`runner.rs` drives core calls, `state.rs` retains Python call state, and
`dispatch.rs` executes core-selected delivery and retains logging arguments
`handle.rs` owns the Created/Running/Suspended/Closed execution protocol
`bindings.rs` invokes Python integrations, and `preparation.rs` projects shared
preparation inputs. Runtime waiting and panic containment stay in `execution.rs`

The Python coroutine driver lives in `litellm/rust_bridge/lifecycle.py`
Read Python state only at core-selected checkpoints. Preserve argument identity,
aliases, omitted values, and deliberate copies across suspension and callbacks

## Data handling

Project only consumed values at their reference read points. Keep native
provider state typed in core and preserve captured upload and request bytes
Do not log OCR documents or upstream bodies, or expose them through raw errors
Measure conversion and copy costs before optimizing large payloads

## Verification

`cargo test --workspace` must compile this crate. Cover disabled, enabled, and
unavailable execution, effect-free decline, terminal errors, callback delivery,
re-entry, GC, and cancellation. Validate the installed extension with a fresh
wheel and positive native execution evidence for lifecycle changes

Keep `_native.pyi` consistent with the exported bindings, including the
Future-returning value bindings and coroutine-returning lifecycle bindings
