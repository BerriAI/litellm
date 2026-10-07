> Hand-off copy of `~/repos/notes/rust-migration/router-poc.md` for cloud sessions that cannot see the local notes vault. Delete this file before opening a PR

# Rust router POC

Agreed 2026-10-06 in an interview session. Branch `litellm_rust_router_poc` (off `main` at `2cee61626d`). Builds on [[router-structure]] and [[router-rust-mapping]] (this is option 2 from there)

## What the POC has to prove

- The Python/Rust boundary design works: who owns state, how kwargs, errors, streams and metadata cross, nothing counted twice
- Behavior parity: the Rust backend makes the same picks, retries and fallbacks as Python and writes the same Redis keys, values and TTLs
- The crate design from [[router-rust-mapping]] (snapshot + `ArcSwap`, filter pipeline, `Operation` enum, execute loop) holds up as real code

Performance is not a goal of the POC

## Python side

`litellm.Router` becomes a thin facade over a backend interface. Two equal-behavior backends sit behind it:
- the current Python implementation, moved behind the interface unchanged
- a Rust backend exposed through `python-bridge`

Backend choice: a new `Route.ROUTER` rule in `litellm/rust_bridge/catalog.py`, toggled by `LITELLM_RUST` like the other routes, resolved once at `Router(...)` construction

Facade surface: **provisional decision, expected to change.** For now the facade exposes the full current `Router` surface, raw internals included (`cache`, `cooldown_cache`, `pattern_router`, `model_list`, index maps, ...), so no caller changes. The Rust backend's stand-ins for those internals live behind one seam, and every place that depends on this choice carries a marker pointing back here. The likely replacement is a typed protocol with real methods, with the ~164 raw-attribute call sites in the proxy migrated to it

## Rust side

- The Rust backend owns the registry: `ArcSwap<Snapshot>` with deployments, indices, aliases, patterns and settings. `upsert_deployment`, `delete_deployment` and `update_settings` go through Rust. Python attributes like `model_list` and `model_names` are read-only views projected from the snapshot
- Runtime state (cooldowns, usage counters, strategy stats) lives in memory plus Redis, with key formats, values and TTLs identical to Python so both backends can share one Redis

## Request flow

Rust drives the loop: resolve, filter, pick, retry, fallback, cooldown

Each attempt goes out as an `Invoke { deployment, operation, kwargs }` host op on the existing `coroutine` protocol. Python answers it inline in the caller's task by calling `litellm.<op>(**kwargs)`. kwargs cross as opaque Python objects; the router only reads the model, the stream flag and the metadata it writes. Native dispatch (`Operation -> Native | Invoke`) is part of the design but nothing uses it yet: every operation is an Invoke in the POC

The coroutine is driven by an async Python driver and a sync Python driver, so sync methods (`router.completion`, ...) are supported too

Python-bound features (auto, complexity, adaptive and quality routers, custom strategies, routing plugins, pre-routing hooks) are host ops that call the existing Python implementations

## Errors

When an Invoke raises, the Python driver classifies the exception into a small typed record (kind, `status_code`, `retry_after`, provider) and sends that to Rust. The original exception object rides along opaquely, so the facade re-raises it unchanged

## Streaming

The Invoke result is a Python stream. The Python driver buffers chunks with a per-format "is this content" check. If the stream fails before the first content chunk, it resumes Rust with `StreamFailedBeforeContent { error }`, and Rust returns the next attempt. Chunks never cross the boundary, so it's one crossing per attempt

## Side effects

Rust returns a typed per-call outcome (deployment, model id, attempted targets, fallback info, routing decision). The Python driver writes it into kwargs and metadata exactly where today's readers look for it. Rust records usage, latency and cooldowns itself, and the Python router callbacks aren't registered for Rust-backed instances, so nothing is counted twice

## Unsupported config

Decided at construction. The Rust backend checks the config up front. If anything unsupported is configured, `RUST_OPT_IN` declines to the Python backend for the whole instance (logged) and `RUST_REQUIRED` raises. Raw members that aren't emulated yet raise `NotImplementedError` naming the member

## First increment

- Core loop: multiple deployments per group, simple-shuffle, retries with backoff, generic fallbacks, cooldowns (memory + Redis), usage counters, non-streaming and streaming calls
- Typed fallbacks and streams: context-window and content-policy fallbacks, retry policies, mid-stream retry before the first content chunk

Later: other strategies, routing groups, filters (aliases, patterns, access groups, tags, health checks), update_settings, native operations

## Parity testing

- Existing behavior tests that go through `Router(...)` get parametrized over both backends. Rust-side failures are the to-do list
- A differential harness: same config, scripted fake Invoke outcomes (errors, latencies), seeded randomness. Assert both backends make the same decisions and write the same Redis keys, values and TTLs

## Order of work

1. Facade refactor: move today's `router.py` behind the backend interface with no behavior change. Its own commit
2. `Route.ROUTER` catalog rule and backend selection at construction
3. Rust crate: snapshot/registry, `Operation`, execute loop, host-op types
4. `python-bridge` export + async/sync Python drivers for the router coroutine
5. Differential harness and parametrized tests, then fill in the first increment until they pass

## Implementation notes

- Each Invoke awaits a small Python helper, not `litellm.<op>` directly. The helper catches the exception and returns a typed outcome (classified record plus the original exception). For streams it also buffers until the first content chunk before returning, so classification and buffering both stay in Python and each attempt is still one host op
- `host-python::run_call` already gives both drivers: async goes through `lifecycle._settle` in the caller's task, sync answers host ops inline. No new driver. The router does not go through `run_public_call`, because the `litellm.<op>` behind each Invoke already runs legacy Logging and preflight
- Deployment normalization and `generate_model_id` stay in Python (they depend on Python `str()` / `json.dumps` output). Rust receives normalized deployments with ids already set and owns the snapshot built from them
- Redis parity risk: Python's sync `RedisCache.set_cache` writes `str(value)` (a Python repr), async paths write `json.dumps`, counters use `INCRBYFLOAT` with a TTL set only when absent, and keys pass through `check_and_fix_namespace`. Parity means matching whichever Python path writes each key. Rust state sits behind a store trait: in-memory first, then Redis
- Keys in scope for the first increment: `deployment:{id}:cooldown` (CooldownCacheValue: exception_received masked to 50 chars, status_code as str, timestamp, cooldown_time; TTL = cooldown time), `deployment:{id}:allowed_fails[:{suffix}]` (fleet-wide counter, TTL = cooldown time), `{id}:successes` / `{id}:fails` (local only, TTL 60)

## Log

- 2026-10-06: design agreed, branch created

## Checkpoint 2026-10-06 (handoff to cloud)

Done: step 1, the facade refactor. `litellm/router.py` is a thin `Router` over `litellm/router_backends/python_router.py` (`PythonRouter`, the unchanged implementation). `Router.backend` holds the backend. Class-level names are forwarded with `_Forwarded` descriptors (so `MagicMock(spec=Router)` and unbound statics like `Router.generate_model_id` keep working). Instance attributes are forwarded with `__getattr__` / `__setattr__` / `__delattr__`. Under `TYPE_CHECKING`, `Router` subclasses `PythonRouter`, which is how the provisional full-surface decision shows up to the type checker

Test migration done in that commit: private helpers and module-global patch strings now target `litellm.router_backends.python_router`; tests that subclass `Router` to override hooks the backend calls on itself (`_InjectedFallbackRouter`, `_TierRouter`, `OldSignatureRouter`) and tests that `patch.object(Router, <internal>)` now use `PythonRouter`; `_live_routers` holds backends, so identity checks use `router.backend`. Subclasses that only override methods called from outside the router (proxy auth tests, `RecordingRouter`) still subclass `Router`

Python behavior to reproduce: see `POC-behavior-spec.md` next to this file

Next: step 2 (`Route.ROUTER` in `litellm/rust_bridge/catalog.py`, `RUST_OPT_IN`, backend chosen in `Router.__init__`), then step 3 (Rust crate). Bridge mechanics to reuse: `host-python::run_call` (not `run_public_call`), `PythonHostCalls::begin_host_call` / `resume_host_call` for the Invoke op (see `python-bridge/src/cache/python/host.rs` for the async-op pattern), `HookChain::new()` for no hooks, `hosted_call` from `litellm-host`. `arc-swap` 1.9.2 is already in `Cargo.lock`
