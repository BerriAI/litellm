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
- 2026-10-07: steps 2, 3 and 4 landed, facade verification running

## Checkpoint 2026-10-07

Done:
- Step 1 verification: in progress (tests on branch vs its base `cb76270b`, then `make check`). Found so far: an isort failure in `python_router.py` and LIT010/LIT008 in `router.py`, both fixed
- Step 2: `Route.ROUTER` (`RUST_OPT_IN`) in `litellm/rust_bridge/catalog.py`. `litellm/router_backends/selection.py::select_backend` runs in `Router.__init__`; `rust_support.py` decides what the Rust backend can serve (declines to `PythonRouter` under opt-in, raises under `RUST_REQUIRED`). `RustRouter` is still a stub, so everything declines
- Step 3: the Rust crate. `engine::Engine::route` runs fallbacks -> retries -> attempt against a `host::RouterHost` (`invoke`, `sleep`). Modules: `snapshot` (registry), `settings`, `failure` (`Classified` record, `Rejection` for router-raised errors), `selection` (resolve, cooldown filter, retry skip, `simple_shuffle`), `retry` (`should_retry_this_error`, backoff), `fallback` (chain lookup), `cooldown` (failure callbacks), `store` (memory + Redis with Python's encodings), `pyrepr` (Python `str()` of dicts), `random` (CPython MT19937), `operation` (`Operation`, `Dispatch`, all `Invoke` today)

Design notes made while implementing:
- Python-bound facts cross inside `Classified`: exception classes in MRO order, `str(e)`, status, stamped `num_retries`, the two Retry-After readings (sleep and cooldown use different header precedence), guardrail/exempt flags, `callbacks_ran` (false for errors the attempt raises after litellm returned, e.g. blocked output) and `exact_litellm_type`
- Metadata and exception writes the router makes (log_retry breadcrumbs, hop bucket copies, exhausted-retry stamps, the fallback debug text) are `host::Op`s the host applies in order before the next attempt or on the final result. Buckets are numbered: 0 is the request's own, each fallback hop opens a copy of its parent's, matching `run_async_fallback`
- Router-raised errors (`RouterRateLimitError`, no-healthy `BadRequestError`) are `Raised::Router { id, rejection }`; the host materializes each id once
- Sleeps are a host op so tests can observe them and the sync driver can block
- Python's shared-logging behavior is kept: with `litellm_logging_obj` passed, only the first failure runs the failure callbacks and fallback-hop failures cool down through `_trigger_cooldown_for_failed_deployment`

- Step 4: `python-bridge` exports `Router` (`src/routes/router/`): `Router(deployments, settings, providers, redis_url, seed)` holds the `Engine`; `route(call, driver, asynchronous)` runs one call through `host-python::run_call` with `HookChain::new()`. `Invoke` and `Sleep` are host ops on `RouterHostCall`; async answers await the driver's coroutine inline, sync answers call `invoke_sync`/`sleep_sync`. `map_error` asks the driver for the exception to raise (ops applied, rejections materialized once per id), `encode_response` asks it for the final response (ops applied, retry/fallback headers added)
- Python: `RustRouter` (`litellm/router_backends/rust_router.py`) builds a `PythonRouter` from the same arguments minus Redis and `discard()`s it, so normalization, `generate_model_id` and attempt kwargs building stay Python and no router callbacks are registered. It projects deployments and resolved settings to the native router. `acompletion`/`completion` are served; streaming, `priority`, `specific_deployment`, request-level `model_group_retry_policy`, `include_fallback_errors`, `_router_weights`, `mock_timeout`, `mock_testing_rate_limit_error` and client-side credentials raise `NotImplementedError`. Other members raise `NotImplementedError` naming the member, except the read views `model_list`, `model_names`, `get_model_names`
- `litellm/router_backends/rust_call.py::RoutedCall` is the per-call driver: it mirrors `_acompletion`/`_completion`'s body after selection (same `PythonRouter` helpers through a typed `AttemptRouter` protocol), classifies exceptions, and applies the router's ops to numbered metadata buckets
- Tests: `tests/unit/router_backends/test_rust_call.py` (classification, buckets, rejections, debug text), `tests/test_litellm_rust/router/test_rust_router.py` (native; same config and seed through both backends: identical pick sequences, headers, error text, cooldown-then-reject sequence, sync path, facade selection)

Next: step 5. Parametrize the existing `Router(...)` behavior tests over both backends (the Rust side's `NotImplementedError`s and mismatches become the to-do list), and grow the differential harness: scripted failures per deployment, a live `redis-server` shared by both backends and compared key by key (values, TTLs). Then streaming (buffer to the first content chunk in the driver) and the generic operations (`aresponses`, `aanthropic_messages`)

Known gaps to close or decline: chat streaming pre-first-chunk failures (Python goes straight to fallbacks, no in-group retry), `include_fallback_errors`, request-level `model_group_retry_policy`/`specific_deployment`/`priority`, CustomLogger fallback-event hooks, `router_cooldown_event_callback`, the 200-key `InMemoryCache` eviction, fallback keys containing `/` (provider-prefixed matching needs the cost map)
