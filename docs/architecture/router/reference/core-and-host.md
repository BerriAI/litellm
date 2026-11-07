# Core, config and host interfaces

The interfaces between the Python facade, the Rust core and the host it runs
under. [The core and the host](../explanation/core-and-hosts.md) explains why
they are shaped this way.

![The Router facade uses the RouterCore protocol; RustCore implements it and
holds a handle to the Rust RustRouter, which calls the Host trait that
PythonHost implements](../img/core-and-host-1.svg)

## Python side

| Name | File | What it is |
| --- | --- | --- |
| `Router` | `litellm/router.py` | the facade. `Router(**arguments)` and `Router.from_config(config, host=python_host)` |
| `RouterCore` | `litellm/router_core.py` | the protocol the facade delegates to |
| `make_core(config, host)` | `litellm/router_core.py` | builds the core once, when the Router is built |
| `RustCore` | `litellm/router_rust/core.py` | holds one `RouterCoreHandle` and answers its steps |
| `RustBridgeDeclined` | `litellm.rust_bridge._native` | raised when the handle cannot read a config |
| `RouterConfig` | `litellm/router_config.py` | the constructor arguments, frozen, with plain containers copied (`build_router_config`) |
| `RouterDynamicConfig` | `litellm/router_config.py` | a versioned snapshot of the process settings the router reads per call (`current_dynamic_config()`) |
| `PythonHost` | `litellm/router_python_host.py` | `invoke(op)` returns the `litellm` endpoint function, plus `sleep`, `dynamic()`, the callback lists, the global cache and `routing_cache(config)` |
| `router_attempt` | `litellm/router_attempt.py` | builds one attempt's keyword arguments for the deployment the core picked |
| `RouterHost` | `litellm/router_host.py` | what only the application can answer (vertex pass-through credentials, member auto-router authorization) |
| `DeprecatedRouterMembers` | `litellm/router_deprecated.py` | the old Router member names, served with a `DeprecationWarning` |

`RouterDynamicConfig` fields are `num_retries`, `max_fallbacks`,
`router_max_fallbacks`, `allowed_fails`, `request_timeout`, `fallbacks`,
`default_fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`,
`drop_params`, `provider_list`, `expose_router_debug_in_errors`,
`redact_user_api_key_info`, `model_alias_map`, `model_cost`,
`known_custom_logger_compatible_callbacks`, `exception_types`, `callbacks`,
`success_callback`, `failure_callback`, `input_callback`, `callback_manager` and
`cache`, plus its `version`.

## The step protocol

`RouterCoreHandle(config, dynamic)` in `litellm.rust_bridge._native` wraps one
Rust `Router`. `start(op, request)`, `select(model, request)` and
`healthy(model, request)` return a `RouterCall`, and so do the report and model
info queries. The driver calls `RouterCall.next(answer)` until it gets `done` or
`failed`. Every step carries the routing events since the last step.

| Step | The core needs | The driver answers with |
| --- | --- | --- |
| `attempt` | one attempt run on the named deployment | the response and its total tokens, or the typed failure |
| `sleep` | a backoff of `seconds` | nothing, after sleeping through the host |
| `allow_fallback` | whether a cross-group fallback `target` is allowed | a bool from the fallback access and budget checks |
| `pre_route` | the sub-routers, session binding or a custom strategy before a pick resolves `model` | a model group, a deployment the custom strategy picked, or `None` |
| `select` | one host stage run on the candidates, named by `stage` | see the stage table below |
| `cache` | a routing cache read or write (`get`, `set`, `batch_get`, `increment` and the rest) | the values read, or an acknowledgement |
| `secrets`, `validate`, `provider`, `added` | one deployment's build, while a config or change set is applied | the resolved secrets, the validated entry, the provider, the added entry |
| `model_info` | a cost map or capability lookup during a model info query | the lookup's documents |
| `done` | nothing more | the result goes back to the caller, with routing headers |
| `failed` | nothing more | the error goes back to the caller, by its `error_kind` |

| `select` stage | The host returns |
| --- | --- |
| `routing_plugins` | the candidate models the routing plugins left, after running them when `run` is set |
| `callback_filters` | the deployment ids the `CustomLogger` filters left, or a failure |
| `plugins` | the deployments `Router(plugins=[...])` objects left |
| `pre_call` | the facts the pre-call checks decide by, such as token counts |
| `tags` | the deployments tag routing left |
| `text_only` | the candidates that declare no vision support |
| `strategy` | the candidates with the pick first, for a strategy the core does not implement |

`apply(changes)` returns the new registry version, `registry_version()` reads
it, and `retire()` retires the router.

## Rust side

All of these live in `litellm-rust/crates/router-core/src/`.

| Type | Module | What it is |
| --- | --- | --- |
| `Router`, `RouterBuilder` | `lib.rs` | `Router::new(config, dynamic, host)`, or the builder with `randomness` and `clock` for tests |
| `RouterConfig`, `RouterDynamicConfig`, `Op`, `RoutingStrategy`, `RetryPolicy`, `AllowedFailsPolicy` | `config.rs` | static and dynamic config, the operation name, the strategies (`is_native` is true for `simple-shuffle`) |
| `Registry`, `Deployment`, `ChangeSet`, `RegistryVersion`, `deployment_id` | `registry.rs` | the immutable, indexed deployments |
| `Settings` | `settings.rs` | the frozen settings view |
| `Health`, `Outcome`, `Clock`, `cooldown_key`, `usage_key` | `health.rs` | cooldowns, allowed fails, usage and the outcome counters |
| `RouterCache`, `InMemoryRouterCache` | `cache.rs` | the routing state store the host provides |
| `Randomness`, `SeededRandomness`, `simple_shuffle` | `select.rs` | the filter chain and `simple-shuffle` |
| `resolve_request`, `all_deployments` | `resolve.rs` | name resolution, a port of `_common_checks_available_deployment` |
| `routing_context`, `request_override` | `strategy.rs` | which strategy a pick uses |
| `RouterWeights`, `validate` | `weights.rs` | per-request weights |
| `RouterResponse` | `call.rs` | the call loop's result with the deployment, group, retries and fallbacks used |
| `StreamPolicy`, `StreamOrigin` | `call/stream.rs` | what may take over a stream that broke |
| `Fallbacks`, `FallbackTarget` | `fallback.rs` | the parsed fallback lists |
| `Host`, `Attempt`, `RouterRequest`, `Output`, `RouterEvent`, `RoutingContext`, `SelectStage` | `host.rs` | the host seam |
| `Error`, `ConfigError`, `Unsupported`, `AttemptError`, `FailureKind` | `error.rs` | errors at build time, at call time, and per attempt |

The other modules hold one stage or query each: `access.rs` (access groups and
windows), `ptu.rs` (PTU shares), `health_routing.rs`, `precall.rs`,
`prerouting.rs`, `routing_groups.rs`, `routing_plugins.rs`, `retry.rs`,
`mock.rs` (`mock_testing_*` failures), `build.rs` (the deployment build),
`lookup.rs` and `host_index.rs` (the indexes the Python surface reads),
`model_info.rs` and `group_info.rs` (model info merges).

`Op` is the operation's name as a string, such as `acompletion`, `aembedding`,
`aresponses` or `aanthropic_messages`. The core passes it to the host and never
interprets it.

### `Host`

| Method | Called for |
| --- | --- |
| `invoke(attempt)` | every attempt. There is no native attempt path |
| `select(stage, ctx, candidates)` | the `Plugins`, `Strategy`, `Tags` and `TextOnly` stages |
| `pre_route(ask)` | pre-routing, and the whole pick for a custom strategy |
| `routing_plugin_models`, `callback_filters`, `pre_call_facts` | the stages that need live Python objects |
| `allow_fallback(request, target)` | the access and budget checks before a cross-group fallback |
| `sleep(duration)`, `observe(event)`, `cache()` | backoff, routing events, the routing state store |
| `secrets`, `validate`, `provider`, `added` | the deployment build |
| `model_info(lookup)` | cost map and capability lookups |
| `supports(op, provider)`, `can_select()` | what the host can serve. Both default to true, and `PythonHost` serves everything |

### `RouterRequest`

| Field | Meaning |
| --- | --- |
| `model` | the requested name, a model group, alias, routing group, deployment id or wildcard match |
| `body` | the request params the core's own checks read |
| `team_id`, `proxy_admin`, `allowed_models`, `admin_model` | the caller facts the team and access filters read |
| `specific_deployment` | the caller named a deployment's `litellm_params.model` |
| `pinned_deployment_id` | a deployment the session or affinity binding pinned |
| `num_retries`, `model_group_retry_policy`, `retry_policy_raises` | per-request retry settings, which win over the router's |
| `disable_fallbacks`, `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, `max_fallbacks` | per-request fallback settings |
| `cooldown_time` | a per-request cooldown |
| `router_weights` | per-team weights, model group to deployment id to weight |
| `routing_strategy` | a per-request strategy override |
| `filter_tags`, `has_image` | inputs to the tag and text-only stages |
| `pass_through` | a pass-through route, which serves only pass-through deployments |
| `target_order`, `excluded_ids`, `retry_skipped_ids` | the order an order fallback targets, and ids the pick must skip |
| `pre_routes`, `unrouted`, `skip_weighted_failover` | whether and how pre-routing and weighted failover run |
| `mock_failure`, `health_check_probe` | test failures and health check probes |
| `payload` | what the host needs to run the call itself, such as the Python kwargs |

### `Attempt` and `Output`

`Attempt` carries `op`, `request`, `deployment`, `name` (the requested name or
the fallback target), `model_group` (the group served, after fallbacks),
`target_order`, `candidate_ids`, `retry` (0 for the first attempt in this
group), `fallback_depth`, `fallback_level` (0 for the requested name),
`timeout`, `params`, `scope`, `max_retries` and `model_group_size`.

`Output` carries the `value` the host hands back unchanged, `total_tokens`,
`blocked` (`ContentFilter` or `Refusal`, which the fallback rules read) and
`deferred`, which is set for a stream or realtime session whose outcome the host
reports when it ends.

### `RouterEvent`

| Event | Fields |
| --- | --- |
| `Selected` | `model_group`, `deployment_id` |
| `Succeeded` | `deployment_id` |
| `AttemptFailed` | `deployment_id`, `kind`, `status` |
| `CooldownSet` | `deployment_id`, `cooldown` |
| `Retry` | `model_group`, `retry`, `delay` |
| `Fallback` | `from`, `to`, `order`, `depth`, `level`, `scope`, `parent`, `max_fallbacks` |
| `FallbackDone` | `from`, `to`, `level`, `scope`, `parent`, `succeeded` |
| `Blocked` | `level`, `model`, `blocked` |
| `RetryLogged` | `scope`, `level`, `model`, `attempted_retries`, `cause` |
| `LevelFailed` | `level`, `model`, `error`, `handling` |
| `MockFailed` | `mock`, `level`, `model`, `num_retries` |
| `RetryPolicyRaised` | `level`, `model`, `cause` |
| `OrderChainFailed` | `level`, `last_hop` |
| `RetriesExhausted` | `model_group`, `attempted`, `max`, `level`, `picked` |

The driver uses the later events to raise the exception a caller expects at that
point, with the retry and fallback attributes set.

### `FailureKind`

The kinds are `RateLimit`, `ContextWindowExceeded`, `ContentPolicyViolation`,
`Authentication`, `PermissionDenied`, `NotFound`, `BadRequest`, `Timeout`,
`ApiConnection`, `InternalServer`, `ServiceUnavailable`, `BadGateway`,
`GuardrailIntervention` (never retried or fallen back from), `Declined` (litellm
rejected the request before calling, and it never cools a deployment down),
`CallerCredential` (the caller's own credential failed) and `Other`.
`AttemptError` adds `status`, `retry_after`, `num_retries`, the provider's error
text for the cooldown record, and `agentic_followup`, which marks an agentic
follow-up call that failed after the first call was served and is never retried.

### Build-time errors

`Router::new` returns `ConfigError` when the registry cannot be built, for
example from a malformed deployment or a duplicate id, or when the host cannot
serve the config. `Unsupported` names what is missing: a strategy or plugin on a
host that cannot select, or an operation and provider pair the host does not
support. Under `PythonHost` neither happens, since it serves every operation and
strategy.
