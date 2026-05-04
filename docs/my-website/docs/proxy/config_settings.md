# Proxy Configuration Settings

This document describes the configuration settings for the LiteLLM Proxy.

## Router Settings

### router_settings - Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| model_list | List[dict] | List of models to route to |
| redis_url | Optional[str] | Redis URL for caching |
| redis_host | Optional[str] | Redis host |
| redis_port | Optional[int] | Redis port |
| redis_password | Optional[str] | Redis password |
| redis_db | Optional[int] | Redis database |
| caching_groups | Optional[List] | Caching groups |
| cache_responses | Optional[bool] | Enable response caching |
| cache_kwargs | Optional[dict] | Cache kwargs |
| cooldown_time | Optional[float] | Cooldown time between retries |
| disable_cooldowns | Optional[bool] | Disable cooldowns |
| num_retries | Optional[int] | Number of retries |
| timeout | Optional[float] | Timeout for requests |
| stream_timeout | Optional[float] | Timeout for streaming |
| retry_after | Optional[float] | Retry after duration |
| retry_policy | Optional[dict] | Retry policy configuration |
| routing_strategy | Optional[str] | Routing strategy |
| routing_strategy_args | Optional[dict] | Routing strategy arguments |
| fallbacks | Optional[List] | Fallback models |
| default_fallbacks | Optional[List] | Default fallback models |
| context_window_fallbacks | Optional[List] | Context window fallbacks |
| content_policy_fallbacks | Optional[List] | Content policy fallbacks |
| allowed_fails | Optional[int] | Allowed failures |
| allowed_fails_policy | Optional[dict] | Allowed fails policy |
| health_check_ignore_transient_errors | Optional[bool] | Ignore transient errors in health check |
| health_check_staleness_threshold | Optional[int] | Health check staleness threshold |
| enable_health_check_routing | Optional[bool] | Enable health check routing |
| enable_pre_call_checks | Optional[bool] | Enable pre-call checks |
| optional_pre_call_checks | Optional[List] | Optional pre-call checks |
| client_ttl | Optional[int] | Client TTL |
| deployment_affinity_ttl_seconds | Optional[int] | Deployment affinity TTL |
| model_group_affinity_config | Optional[dict] | Model group affinity config |
| model_group_alias | Optional[dict] | Model group alias |
| model_group_retry_policy | Optional[dict] | Model group retry policy |
| assistants_config | Optional[dict] | Assistants config |
| provider_budget_config | Optional[dict] | Provider budget config |
| tag_filtering_match_any | Optional[List] | Tag filtering match any |
| enable_tag_filtering | Optional[bool] | Enable tag filtering |
| alerting_config | Optional[dict] | Alerting config |
| default_litellm_params | Optional[dict] | Default litellm params |
| default_max_parallel_requests | Optional[int] | Default max parallel requests |
| default_priority | Optional[int] | Default priority |
| debug_level | Optional[str] | Debug level |
| set_verbose | Optional[bool] | Set verbose |
| ignore_invalid_deployments | Optional[bool] | Ignore invalid deployments |
| max_fallbacks | Optional[int] | Max fallbacks |
| polling_interval | Optional[float] | Polling interval |
| search_tools | Optional[List] | Search tools |
| fetch_tools | Optional[List] | Fetch tools |
| guardrail_list | Optional[List] | Guardrail list |
| router_general_settings | Optional[dict] | Router general settings |

### general_settings - Reference

## Other settings...
