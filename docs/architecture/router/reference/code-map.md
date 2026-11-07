# Code map

Where the router's code lives. Paths are relative to the repo root.

![The router code map: the Python facade and its modules lead to the RustCore
step driver, the router_rust step modules, the python-bridge crate and the
router-core crate](../img/code-map-1.svg)

## Python

| File | Holds |
| --- | --- |
| `litellm/router.py` | `Router`, the facade. It builds the config, chooses the core and delegates every member |
| `litellm/router_api.py` | the entry point views and types, see [Entry points](entry-points.md), plus `ROUTER_INIT_ARGS` and `apply_changes` |
| `litellm/router_core.py` | the `RouterCore` protocol and `make_core` |
| `litellm/router_config.py` | `RouterConfig`, `RouterDynamicConfig`, and the only router reads of `litellm.*` settings |
| `litellm/router_deprecated.py` | `DeprecatedRouterMembers`, the old member names with a `DeprecationWarning` |
| `litellm/router_python_host.py` | `PythonHost`, the host inside a Python process: endpoint functions, sleep, dynamic config, callbacks, caches |
| `litellm/router_host.py` | `RouterHost`, what only the application can answer, and `litellm/proxy/_router_host.py` with the proxy's version |
| `litellm/router_strategy/` | the stateful strategies, `lar1`, the sub-routers, the budget limiter and tag routing. The core reaches them through the select and pre-route steps |
| `litellm/router_utils/` | helpers the host uses, such as the cooldown cache, the health state cache, pre-call checks and the response headers |
| `litellm/litellm_core_utils/` | helpers the router and other modules share, such as wildcard matching, PTU shares and the retry and fallback headers |
| `litellm/types/router.py` | `Deployment`, `RoutingGroup`, `RetryPolicy` and the other config types |
| `litellm/proxy/route_llm_request.py` | maps a proxy request to a Router endpoint method |

### `litellm/router_rust/`

Everything the core needs on the Python side.

| Path | Holds |
| --- | --- |
| `core.py` | `RustCore`, the step driver for async and sync calls |
| `steps/` | answers to the steps that need live Python objects: `selection.py` (strategies), `prerouting.py` (sub-routers, session binding, custom strategies), `precall.py` (pre-call facts), `plugins.py` (`Router(plugins=[...])`), `callbacks.py` (`CustomLogger` filters), `access.py` (caller access facts) |
| `invoke/` | one attempt: `endpoints.py` (the endpoint wrappers), `shaping.py` (request and response shaping), `streaming.py` (stream wrappers that report the outcome when a stream ends), `slots.py` (`max_parallel_requests`), `walk.py` (retry and fallback metadata), `prompt_management.py`, `websocket.py` |
| `surface/` | the Python `Router` API over the core's state: `registry.py`, `model_list.py`, `indexes.py`, `members.py`, `model_info.py`, `settings.py`, `health.py`, `health_routing.py`, `retries.py`, `fallbacks.py`, `alerting.py`, `replay.py` |

`litellm/router_attempt.py` builds the keyword arguments of one attempt for the
deployment the core picked. It does not pick, retry or record anything.

## Rust

`litellm-rust/crates/router-core/src/`.

| File | Holds |
| --- | --- |
| `lib.rs` | `Router`, `RouterBuilder`, and the crate overview |
| `config.rs` | `RouterConfig`, `RouterDynamicConfig`, `Op`, `RoutingStrategy`, retry and allowed fails policies |
| `registry.rs` | `Registry`, `Deployment`, `ChangeSet`, `deployment_id` |
| `build.rs` | building deployments from config with the host's `secrets`, `validate`, `provider` and `added` steps |
| `resolve.rs` | name resolution, a port of `_common_checks_available_deployment` |
| `select.rs` | the filter chain and `simple-shuffle` |
| `access.rs`, `ptu.rs`, `health_routing.rs`, `precall.rs` | the access group and window, PTU share, health check and pre-call filters |
| `prerouting.rs`, `routing_plugins.rs`, `routing_groups.rs` | pre-routing, routing plugin candidates and routing group members |
| `strategy.rs`, `weights.rs` | which strategy a pick uses, and per-request weights |
| `call.rs`, `call/stream.rs` | the call loop with retries, fallbacks and weighted failover, and stream recovery |
| `retry.rs`, `fallback.rs`, `mock.rs` | retry rules, the fallback lists, and `mock_testing_*` failures |
| `health.rs` | cooldowns, allowed fails, usage, and the cache keys |
| `cache.rs` | `RouterCache` and an in-memory implementation for tests |
| `settings.rs` | the frozen `Settings` view |
| `host.rs` | `Host`, `Attempt`, `RouterRequest`, `RouterEvent`, `RoutingContext` |
| `lookup.rs`, `host_index.rs` | the indexes and listings the Python surface reads |
| `model_info.rs`, `group_info.rs` | model info and model group info merges |
| `error.rs` | `ConfigError`, `Error`, `AttemptError`, `FailureKind` |

`litellm-rust/crates/python-bridge/src/router/` exposes the core to Python as
`RouterCoreHandle` and `RouterCall`, turns Python's config and dynamic snapshot
into the core's types, and turns each hostcall into a step.

## Tests and checks

| Path | What it covers |
| --- | --- |
| `tests/unit/test_router/core/` | the core suite, run against the core with a scripted host: selection, retries and cooldowns, fallbacks, registry change sets, endpoints |
| `litellm-rust/crates/router-core` unit tests | the Rust modules, with seeded randomness and a test clock |
| `tests/unit/test_router/` | router unit tests through the facade, including `test_router_entry_points.py` |
| `tests/code_coverage_tests/router_surface_check.py` | the shrink-only surface allowlist, run in CI |
| `tests/code_coverage_tests/router_litellm_globals_check.py` | checks that only `litellm/router_config.py` reads `litellm.*` settings |
| `tests/code_coverage_tests/router_code_coverage.py` | checks that every router function has a test |
| litefuzz `translation-diff` suite | differential fuzzing of routing behavior, run on every change to the core |
