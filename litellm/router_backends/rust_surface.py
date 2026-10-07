"""Which `Router` members the Rust backend answers without handing the instance over to Python.

Requests it routes itself are methods on `RustRouter`. The names below are answered by the
Python-side router it keeps for normalization, which runtime changes keep in step with the
native snapshot, so its model list, settings and config-derived lookups are the ones Rust
routes by. Cooldown reads there see Rust's cooldowns once the router shares the proxy's Redis.
Any other member hands the instance over to `PythonRouter` for good.
"""

from typing import Final

VIEWS: Final = frozenset(
    {
        "_are_all_deployments_blocked",
        "_get_client",
        "_get_model_from_alias",
        "_get_router_deployment_budget_limiter",
        "_is_deployment_blocked",
        "_is_model_access_group_for_wildcard_route",
        "_routing_groups",
        "adaptive_routers",
        "auto_routers",
        "background_health_check_model_groups",
        "cache",
        "cached_deployment_model_info",
        "cached_model_group_info",
        "complexity_routers",
        "config_deployments",
        "cooldown_time",
        "default_deployment",
        "deployment_latency_map",
        "deployment_names",
        "deployments_for_request",
        "enable_health_check_routing",
        "fallback_access_check",
        "fallback_budget_check",
        "fallbacks",
        "generate_model_id",
        "get_available_guardrail",
        "get_candidate_model_ids_for_route",
        "get_configured_display_name",
        "get_configured_mode",
        "get_configured_service_tiers",
        "get_credential_deployment",
        "get_deployment",
        "get_deployment_by_model_group_name",
        "get_deployment_credentials",
        "get_deployment_credentials_with_provider",
        "get_deployment_model_info",
        "get_discovered_model_info",
        "get_fully_blocked_model_names",
        "get_model_access_groups",
        "get_model_group_info",
        "get_model_ids",
        "get_model_info",
        "get_model_list",
        "get_model_list_from_model_alias",
        "get_model_listing_info",
        "get_remaining_model_group_usage",
        "get_routable_upstream_model",
        "guardrail_list",
        "has_model_id",
        "health_check_ignore_transient_errors",
        "health_state_cache",
        "is_recognized_model",
        "map_team_model",
        "model_group_alias",
        "model_name_to_deployment_indices",
        "num_retries",
        "pattern_router",
        "provider_budget_config",
        "quality_routers",
        "resolve_model_name_from_model_id",
        "resolved_litellm_models",
        "retry_after",
        "routable_model_group",
        "router_general_settings",
        "routing_plugins",
        "search_tools",
        "should_include_deployment",
        "team_model_to_deployment_indices",
        "team_pattern_routers",
        "team_public_model_names",
        "timeout",
    }
)

ASSIGNABLE: Final = frozenset({"cache_responses", "guardrail_list", "search_tools"})
