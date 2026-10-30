# Entry points

`litellm.Router` exposes its endpoint methods plus the entry points below, and nothing else. `router_surface_check.py` enforces that in CI. The views and types are in `litellm/router_api.py`. The facade in `litellm/router.py` delegates each entry point to its `RouterCore`, and the Rust core answers it. The "Was" columns name the Router members each query replaced, which were removed with `DeprecatedRouterMembers`, see [Port code written against the old Router members](../how-to/port-old-router-code.md)

```mermaid
classDiagram
    class Router {
        +acompletion() and the other endpoint methods
        +registry() RouterRegistry
        +settings() RouterSettings
        +health() RouterHealth
        +select(model, request_kwargs, kind) deployment
        +apply(ChangeSet) ApplyResult
        +aapply(ChangeSet) ApplyResult
        +observe(logging_obj, request_scoped) RoutingObservers
        +register_plugin(CustomRoutingStrategyBase)
        +retire()
    }
    class RouterRegistry {
        +version int
        +deployment(id)
        +deployments(model_name, source)
        +names(model_name, kind)
        +resolve(name, kind)
        +group(name, field)
        +access_groups()
        +credentials(id, form)
        +plugins(kind)
        +routing_group(name)
        +client(deployment)
        +aliases() patterns() routing_groups() index()
    }
    class RouterSettings {
        <<frozen, static>>
        +routing_strategy
        +fallbacks ...
        +is_set(name)
        +as_dict()
        +allowed_fails_for(exception)
        +guardrail(name)
    }
    class RouterHealth {
        +store routing cache
        +cooldowns
        +health_checks
        +latency
        +is_blocked(d) all_blocked(ds)
        +usage(model_group, kind)
        +fully_blocked_model_names()
        +failures_this_minute(id)
        +record_failure(id, exc, status)
    }
    class ChangeSet {
        +assign
        +settings
        +upsert
        +delete
        +redis_cache
        +refresh_model_info
    }
    Router --> RouterRegistry
    Router --> RouterSettings
    Router --> RouterHealth
    Router ..> ChangeSet : apply
```

## `registry()` returns `RouterRegistry`

Read-only queries over the configured deployments. The core keeps the registry as an immutable, versioned snapshot, and each query reads the current one

| Query | Returns | Was |
| --- | --- | --- |
| `version` | an int that goes up on every deployment change | |
| `deployment(id)` | `Deployment` or None | `get_deployment` |
| `has_deployment(id)` | bool | `has_model_id` |
| `deployment_dict(id)` | the deployment as its model_list dict | `get_model_info` |
| `deployment_model_info(id, model_name)` | model info merged with the pricing map | `get_deployment_model_info` |
| `router_model_info(...)`, `discovered_model_info(id)`, `zero_cost()` | routing model info, provider-discovered limits, the zero-cost lookup | `get_router_model_info`, `get_discovered_model_info`, `_zero_cost_cache` |
| `deployments(name, source="group")` | the group, its aliases, routing groups, then wildcards | `get_model_list` |
| `deployments(name, source="all")` | every deployment of the group | `get_all_deployments` |
| `deployments(source="config")` | deployments from config, not the DB | `config_deployments` |
| `deployments(name, source="routable", request=, call_type=)` | deployments that pass the routing checks for a request | `get_routable_deployments` |
| `deployments(name, source="first_in_group")` | the first deployment of a group | `get_deployment_by_model_group_name` |
| `deployments(name, source=...)` | also `named`, `alias`, `routing_groups`, `routed_group`, `wildcard`, `for_request` | see the docstring |
| `all()` | the live model list | `model_list` |
| `names(kind="groups")` | every routable model name | `get_model_names` |
| `names(kind="group_set")` | the set of configured group names | `model_names` |
| `names(name, kind="ids")` | deployment ids, of one group or all | `get_model_ids` |
| `names(kind="team_public", team_id=)` | the public model names of a team's deployments | `team_public_model_names_of` |
| `names(kind=...)` | also `deployment_names`, `litellm_models`, `litellm_base_models`, `wildcard`, `route_candidates` | see the docstring |
| `resolve(name, kind="group")` | the model group a name routes to | `routable_model_group` |
| `resolve(name, kind="alias")` | the alias target, or None | `get_model_from_alias` |
| `resolve(name, kind="team_model", team_id=)` | a team's public name to its group | `map_team_model` |
| `resolve(name, kind="recognized")` | whether the router knows the name | `is_recognized_model` |
| `resolve(id, kind="group_of_id")` | the group of a deployment id | `get_model_group` |
| `resolve(name, kind=...)` | also `served`, `from_id`, `upstream`, `wildcard_access_group`, `team_model_alternatives`, `routing_group_alternatives`, `search_tool_support` | see the docstring |
| `group(name, field="info")` | merged `ModelGroupInfo` | `get_model_group_info` |
| `group(name, field=...)` | also `cached_info`, `display_name`, `mode`, `service_tiers`, `token_limits`, `listing_info` | see the docstring |
| `access_groups(...)` | access group name to model names | `get_model_access_groups` |
| `credentials(id, form="params")` | a deployment's credentials | `get_deployment_credentials` |
| `plugins(kind)` | `auto`, `complexity`, `quality`, `adaptive` sub-routers, `routing` plugins, `budget_limiter`, `prompt_caching_pins` | `auto_routers` and friends |
| `routing_group(name)` | the routing group callable as that name, or None | `get_routing_group` |
| `client(deployment, client_type=None)` | the SDK client the router cached for a deployment, or None | `_get_client` |
| `aliases()`, `patterns()`, `routing_groups()`, `index(name)` | the alias map, the wildcard patterns, routing groups, the name index | the old attributes |

Module-level helpers in the same file need no Router

| Function | Returns |
| --- | --- |
| `deployment_id(model_group, litellm_params)` | the stable id of a deployment that has none, the same hash the core uses (was `Router.generate_model_id`) |
| `deployment_usable_by_team(deployment, team_id)` | whether a team may use a deployment |
| `router_init_args()` | the arguments `Router()` accepts (was `Router.get_valid_args`) |
| `fallback_chain(fallbacks, model_group)`, `request_fallback_chain(fallbacks, request_kwargs, model_group)` | the fallbacks configured for a group, or for one request, and the index of the `*` entry |
| `fallbacks_disabled(request_kwargs)` | whether the request turned fallbacks off |
| `references_provider_scoped_resource(request_kwargs)` | whether the request names a file, batch or fine-tuning job only its own provider knows |
| `deployment_metadata(request_kwargs)` | the metadata bucket the router stamped the picked deployment into |
| `mark_advisor_orchestration_failure(exception)` | tags an advisor orchestration failure, so the router does not cool down the deployment it picked |

## `settings()` returns `RouterSettings`

A frozen dataclass with one field per readable setting, listed in `SETTING_FIELDS`. It is built from the router's static `RouterConfig` and the current dynamic snapshot, and it changes only through `apply()`. A setting the router never set reads as None. Treat the field values as read-only

`routing_strategy`, `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, `default_deployment`, `default_litellm_params`, `default_max_parallel_requests`, `cooldown_time`, `allowed_fails`, `allowed_fails_policy`, `disable_cooldowns`, `enable_pre_call_checks`, `enable_health_check_routing`, `health_check_ignore_transient_errors`, `background_health_check_model_groups`, `enable_tag_filtering`, `tag_filtering_match_any`, `tag_routing_prefix`, `router_general_settings`, `provider_budget_config`, `provider_default_deployment_ids`, `fallback_access_check`, `fallback_budget_check`, `kubernetes_pod_discovery`, `search_tools`, `guardrail_list`, `cache_responses`

| Method | Returns | Was |
| --- | --- | --- |
| `is_set(name)` | whether the router has this setting at all | |
| `as_dict()` | the settings as plain values | `get_settings` |
| `allowed_fails_for(exception)` | allowed fails for an error under `allowed_fails_policy` | `get_allowed_fails_from_policy` |
| `guardrail(name)` | a configured guardrail | `get_available_guardrail` |

## `health()` returns `RouterHealth`

| Member | Returns | Was |
| --- | --- | --- |
| `store` | the routing cache, see [Routing state and cache keys](state-and-cache-keys.md) | `cache` |
| `cooldowns` | active cooldowns, `await cooldowns.async_get_active_cooldowns(model_ids=, parent_otel_span=)` | `cooldown_cache` |
| `health_checks` | background health check results | `health_state_cache` |
| `latency` | latency samples by deployment | `deployment_latency_map` |
| `is_blocked(deployment)`, `all_blocked(deployments)` | whether an admin paused them | `is_deployment_blocked`, `are_all_deployments_blocked` |
| `fully_blocked_model_names()` | groups with every deployment paused | `get_fully_blocked_model_names` |
| `await fully_unhealthy_model_names()` | groups with every deployment failing health checks | `async_get_fully_unhealthy_model_names` |
| `await usage(group, kind="current")` | tpm and rpm this minute | `get_model_group_usage` |
| `await usage(group, kind="remaining")` | what is left of the group's limits | `get_remaining_model_group_usage` |
| `team_ptu_ceiling(team_id, group)` | a team's PTU share ceiling | `team_ptu_ceiling` |
| `split_priority(model, kwargs)` | the request priority and the kwargs without it | `split_request_priority` |
| `failures_this_minute(id)` | failures counted against a deployment this minute | `get_deployment_failures_for_current_minute` |
| `record_failure(id, exception, status)` | counts a failure the router did not see itself, cools the deployment down once it is over its allowed fails, and says whether it did | `increment_deployment_failures_for_current_minute` then `set_cooldown_deployments` |
| `priority_header_fields(model, kwargs, header)` | the request fields a priority header sets | `priority_header_update` |

## `select(model, request_kwargs=None, *, kind="pick", ...)`

Runs selection in the core without calling the provider. `kind="pick"` returns one deployment dict, exactly what a real request would get (was `async_get_available_deployment`). `kind="healthy"` returns the filtered candidates before the strategy (was `async_get_healthy_deployments`). `kind="pass_through"` picks among deployments with `use_in_pass_through=True`. Optional keyword arguments are `messages`, `input`, `specific_deployment`, `parent_otel_span` and `health_check_probe`

## `apply(changes)` and `aapply(changes)`

The only way to change a Router after it is built. The core builds the next registry and settings from the whole `ChangeSet`, validates them, and swaps them in at once, so a request sees either none of a change set or all of it. A change set the core rejects raises and changes nothing. The fields

| Field | Effect | Was |
| --- | --- | --- |
| `assign` | replace settings as they are. Names from `SETTING_FIELDS`, `PLAIN_SETTING_ATTRIBUTES` (`num_retries`, `max_fallbacks`, `retry_after`, `timeout`) or `ASSIGNABLE_ALIASES` (`model_group_alias`) | `router.<field> = value` |
| `settings` | validated runtime settings | `update_settings(**settings)` |
| `upsert` | add or replace deployments, by `model_info.id` | `upsert_deployment` |
| `delete` | remove deployments by id | `delete_deployment` |
| `redis_cache` | point routing state at a new Redis cache | `update_redis_cache` |
| `refresh_model_info` | re-read discovered model info, `aapply` only | `arefresh_model_info` |

Both return `ApplyResult(version, upserted, deleted)`, with the registry version after the change and, per upsert or delete, the deployment that was there before

## Lifecycle

`observe(logging_obj=None, *, request_scoped=False)` returns the router's `RoutingObservers`, which receive the core's routing events. With `request_scoped=True` the router stops registering its callbacks globally, and passing a request's `logging_obj` binds them to that request. `register_plugin(strategy)` puts a `CustomRoutingStrategyBase` in place of the configured strategy, run through the host's select bridge. `retire()` releases the router's global callback registrations and routing state, for a Router that is being replaced (was `discard`). Calls on a retired router fail

## `RoutingContext`

What a routing plugin receives instead of the Router. It has `registry()`, `settings()` and `health()`, plus `select`, `observers`, `belongs_to`, `explicit_allowed_fails`, `pre_call_checks`, `count_pre_call_check_tokens`, `update_kwargs_before_fallbacks`, `pre_routing_hook` and `log_retry`. The Rust core gives its own plugins the same shape, a `RoutingContext` with the registry, settings, health, the request, the operation and the model group
