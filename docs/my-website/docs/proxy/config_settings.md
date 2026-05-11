# Proxy config settings
#
# Note: LiteLLM's public docs are maintained in a separate repository.
# This file exists to keep in-repo reference checks in sync with Router settings.

## Router

### router_settings - Reference

| key | description
|---|---
| alerting_config | Router alerting configuration.
| allowed_fails | Number of times a deployment can fail before being added to cooldown.
| allowed_fails_policy | Allowed fails policy configuration.
| assistants_config | Assistants/Responses-related configuration.
| cache_kwargs | Additional kwargs to pass to the Router cache backend.
| cache_responses | Enable caching of responses.
| caching_groups | Configure caching across model groups.
| client_ttl | Time-to-live for cached clients (seconds).
| content_policy_fallbacks | Fallback behavior for content policy errors.
| context_window_fallbacks | Fallback behavior for context window limit errors.
| cooldown_time | Cooldown time (seconds) after a deployment failure.
| debug_level | Debug level for Router logging.
| default_fallbacks | Default fallback models/deployments.
| default_litellm_params | Default parameters applied to completion calls.
| default_max_parallel_requests | Default max parallel requests (scheduler).
| default_priority | Default request priority (scheduler).
| deployment_affinity_ttl_seconds | TTL (seconds) for user-key -> deployment affinity mapping.
| disable_cooldowns | Disable deployment cooldown behavior.
| enable_health_check_routing | Enable health-check-aware routing.
| enable_pre_call_checks | Enable pre-call checks filtering deployments (e.g. context window checks).
| enable_tag_filtering | Enable tag-based deployment filtering.
| fallbacks | Model fallback configuration.
| guardrail_list | Guardrails configuration list.
| health_check_ignore_transient_errors | Ignore transient errors in health checks.
| health_check_staleness_threshold | Threshold (seconds) for considering health checks stale.
| ignore_invalid_deployments | Ignore invalid deployments instead of raising errors.
| max_fallbacks | Maximum number of fallbacks to attempt.
| model_group_affinity_config | Model group affinity configuration.
| model_group_alias | Alias mapping for model groups.
| model_group_retry_policy | Retry policy overrides per model group.
| model_list | List of model deployments.
| num_retries | Number of retries for failed requests.
| optional_pre_call_checks | Optional pre-call checks configuration.
| polling_interval | Polling interval for scheduler completion (seconds).
| prompt_prefix_affinity_min_tokens | Minimum canonical prompt token count before prompt-prefix affinity applies (canonical JSON tokenization, not raw prompt tokens).
| prompt_prefix_affinity_tokens | Number of canonical prompt-prefix tokens used for deterministic prompt-prefix routing (canonical JSON tokenization, not raw prompt tokens).
| provider_budget_config | Provider budget configuration.
| redis_db | Redis DB index.
| redis_host | Redis host.
| redis_password | Redis password.
| redis_port | Redis port.
| redis_url | Redis URL.
| retry_after | Minimum time to wait before retrying a failed request (seconds).
| retry_policy | Retry policy configuration.
| router_general_settings | Router general settings object.
| routing_groups | Routing groups configuration.
| routing_strategy | Routing strategy used for the implicit default group.
| routing_strategy_args | Arguments for the routing strategy (e.g. latency window).
| search_tools | Search tools configuration.
| set_verbose | Enable verbose logging.
| stream_timeout | Stream timeout (seconds).
| tag_filtering_match_any | Tag filtering mode (match any vs match all).
| timeout | Request timeout (seconds).

### router_settings - Notes

`prompt_prefix_affinity_tokens` and `prompt_prefix_affinity_min_tokens` are measured
against tokenization of the canonical JSON representation constructed for the request
(field names, separators, quotes, etc.), not the provider-specific "raw prompt token"
count after rendering the final prompt.

