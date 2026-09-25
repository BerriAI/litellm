# 03 Routing and reliability

How the gateway picks a deployment for a request and what happens when a deployment fails: routing strategies, fallbacks, retries, cooldowns, timeouts, circuit breakers, tag and session routing, the auto router, prioritization, mirroring, and the Rust data plane. Model registration lives in [02-models-providers.md](02-models-providers.md), rate-limit and budget enforcement in [05-budgets-ratelimits-spend.md](05-budgets-ratelimits-spend.md), and what happens inside a single endpoint call in [01-llm-endpoints.md](01-llm-endpoints.md)

## Routing strategies

### routing.strategies: Routing strategies (simple-shuffle, least-busy, latency-based, usage-based, cost-based, rate-limit-aware)
surfaces: sdk, config, ui | flags: redis(some)
docs: https://docs.litellm.ai/docs/routing, https://docs.litellm.ai/docs/proxy/load_balancing
code: `litellm/router.py` (`Router`, `routing_strategy`), `litellm/types/router.py` (`RouterGeneralSettings`)
tests: `tests/test_litellm/test_router.py`, `tests/test_litellm/router_utils/`, `tests/e2e/router/`
registry: reliability.yaml reliability.routing.*
verify: set `routing_strategy: latency-based-routing` in router_settings, send several /chat/completions to a two-deployment model group, and check /model/metrics shows skew toward the faster deployment

### routing.tag_routing: Tag-based routing
surfaces: config, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/tag_routing
code: `litellm/router.py` (`enable_tag_filtering`), `litellm/router_utils/routing_groups.py`, `litellm/types/router.py`
tests: `tests/test_litellm/router_utils/`
registry: reliability.yaml reliability.routing.tagged_marker.*
verify: tag one deployment `tags: ["eu"]`, set `enable_tag_filtering: true`, and call /chat/completions with metadata tags ["eu"]; only the eu deployment serves it

### routing.provider_budget_routing: Provider budget routing
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/provider_budget_routing
code: `litellm/router.py`, `litellm/proxy/spend_tracking/spend_management_endpoints.py` (GET `/provider/budgets`)
tests: `tests/test_litellm/router_utils/`
registry: none
verify: set `router_settings.provider_budget_config` with a per-provider limit, exhaust it with calls, and confirm traffic moves to the next provider

### routing.wildcard_routing: Wildcard routing
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/wildcard_routing
code: `litellm/router.py`, `litellm/router_utils/pattern_match_deployments.py`
tests: `tests/test_litellm/test_router.py`
registry: none
verify: register `openai/*` as a model_name and call /chat/completions with a model name that has no explicit deployment

### routing.auto_router: Auto router (semantic, benchmark, lite autoroute)
surfaces: config, ui, api, cli | flags: none
docs: https://docs.litellm.ai/docs/auto_router/index, https://docs.litellm.ai/docs/auto_router/setup, https://docs.litellm.ai/docs/proxy/auto_routing, https://docs.litellm.ai/docs/learn/autorouter_cli
code: `litellm/proxy/management_endpoints/auto_router_endpoints.py` (`/auto_router/*`), `litellm/router_utils/auto_router_model_naming.py`
tests: `tests/test_litellm/router_utils/` (auto router related files)
registry: reliability.yaml reliability.routing.semantic_auto_router.*
verify: configure an `auto_router` model group per the setup doc, POST /auto_router/test_routing with a prompt, and see which tier it selects

### routing.adaptive_router: Adaptive router
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/adaptive_router
code: `litellm/router.py`, `litellm/types/router.py` (`adaptive_router_config`), `litellm/proxy/proxy_server.py` (GET `/adaptive_router/state`)
tests: `tests/test_litellm/router_utils/`
registry: none
verify: enable `adaptive_router` in router_settings, run traffic, then GET /adaptive_router/state to see learned preferences

### routing.fusion: Fusion (multi-model)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/fusion
code: `litellm/router.py`, `litellm/router_utils/`
tests: `tests/test_litellm/router_utils/`
registry: none
verify: configure a fusion model group per the doc, send one completion, and confirm multiple deployments are invoked and merged

### routing.routing_plugins: Routing plugins
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/routing_plugins
code: `router_plugins.json`, `litellm/router.py` (plugin hooks), `litellm/router_utils/router_callbacks/`
tests: `tests/test_litellm/router_utils/`
registry: none
verify: enable a routing plugin from router_plugins.json in config, run traffic, and confirm the plugin's routing decision in the debug log

### routing.sticky_sessions: Session affinity / prompt-cache routing
surfaces: config, api | flags: redis
docs: https://docs.litellm.ai/docs/auto_router/prompt_caching, https://docs.litellm.ai/docs/tutorials/claude_code_prompt_cache_routing
code: `litellm/router_utils/prompt_caching_cache.py`, `litellm/router.py` (session id routing)
tests: `tests/test_litellm/router_utils/` (prompt caching routing files)
registry: none
verify: send two multi-turn chat completions with the same `litellm_session_id` and confirm both land on the same deployment (check /model/metrics)

### routing.scheduler: Request prioritization / scheduler
surfaces: config, api | flags: none
docs: https://docs.litellm.ai/docs/scheduler
code: `litellm/scheduler.py`, `litellm/proxy/proxy_server.py` (POST `/queue/chat/completions`)
tests: `tests/test_litellm/` (scheduler related files)
registry: none
verify: enable the scheduler in config, POST /queue/chat/completions under load, and confirm prioritized requests are dequeued first

### routing.traffic_mirroring: Traffic mirroring
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/traffic_mirroring
code: `litellm/proxy/proxy_server.py`, `litellm/router.py`
tests: `tests/test_litellm/proxy/` (mirror related files)
registry: none
verify: configure a mirror deployment in model_info, run one call, and confirm the shadow deployment also received the request (no response merge)

## Failure handling

### routing.fallbacks: Fallbacks (model, context_window, content_policy)
surfaces: sdk, config, api, ui | flags: none
docs: https://docs.litellm.ai/docs/routing, https://docs.litellm.ai/docs/tutorials/fallbacks, https://docs.litellm.ai/docs/tutorials/model_fallbacks, https://docs.litellm.ai/docs/proxy/fallback_management
code: `litellm/router.py` (fallback chains), `litellm/router_utils/fallback_event_handlers.py`, `litellm/proxy/management_endpoints/fallback_management_endpoints.py` (`/fallback` CRUD)
tests: `tests/test_litellm/test_router.py`, `tests/e2e/router/`
registry: reliability.yaml reliability.fallback.*, mgmt.yaml mgmt.fallback_management.*
verify: point the primary deployment at a dead api_base, list fallbacks in `router_settings.fallbacks`, and confirm /chat/completions still returns 200 from the fallback

### routing.retries: Retries and retry policy
surfaces: sdk, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/reliability, https://docs.litellm.ai/docs/completion/reliable_completions
code: `litellm/router.py` (`num_retries`, `retry_policy`), `litellm/router_utils/get_retry_from_policy.py`, `litellm/main.py` (`acompletion_with_retries`)
tests: `tests/test_litellm/test_router.py`, `tests/e2e/router/`
registry: reliability.yaml reliability.retry.*
verify: set `num_retries: 2` on a model list entry pointed at a flaky upstream and confirm the call succeeds after retrying

### routing.cooldowns: Deployment cooldowns and allowed_fails
surfaces: sdk, config | flags: none
docs: https://docs.litellm.ai/docs/routing, https://docs.litellm.ai/docs/proxy/reliability
code: `litellm/router_utils/cooldown_handlers.py`, `litellm/router_utils/cooldown_cache.py`, `litellm/router.py` (`allowed_fails`, `cooldown_time`)
tests: `tests/test_litellm/router_utils/` (cooldown files), `tests/e2e/router/`
registry: reliability.yaml reliability.cooldown.*
verify: force a deployment to 429 (tiny rpm), watch it cool down in /model/metrics, and confirm traffic resumes after `cooldown_time`

### routing.timeouts: Timeouts and stream timeouts
surfaces: sdk, config, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/timeout
code: `litellm/router.py` (`timeout`), `litellm/proxy/proxy_server.py` (`request_timeout`), `litellm/litellm_core_utils/streaming_handler.py` (stream timeout)
tests: `tests/test_litellm/` (timeout related files)
registry: reliability.yaml reliability.cooldown.timeout.*, reliability.retry.timeout.*
verify: set `timeout: 1` on a slow deployment and confirm the call returns a timeout error within ~1s and (with fallbacks) still answers

### routing.circuit_breaker: Circuit breaker
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/reliability
code: `litellm/router_utils/cooldown_handlers.py`, `litellm/router_utils/handle_error.py`
tests: `tests/test_litellm/router_utils/`, `tests/e2e/router/`
registry: reliability.yaml reliability.circuit_breaker.*
verify: trip the breaker with repeated failures on one deployment and confirm subsequent requests skip it immediately without waiting

## Deployment surfaces

### routing.key_team_router_settings: Per key / team router settings
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/keys_teams_router_settings
code: `litellm/proxy/management_endpoints/key_management_endpoints.py`, `litellm/proxy/management_endpoints/team_endpoints.py` (router_settings fields on keys/teams)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: mgmt.yaml mgmt.router_settings.*
verify: POST /team/update with a `router_settings` override on a team, then call with that team's key and confirm the override applies

### routing.router_settings_ui: Router settings page
surfaces: ui, api | flags: db
docs: https://docs.litellm.ai/docs/proxy/keys_teams_router_settings
code: `ui/litellm-dashboard/src/app/(dashboard)/router-settings/page.tsx`, `litellm/proxy/proxy_server.py` (`/config/field/update`, `/config/field/info`)
tests: `ui/litellm-dashboard/` (router settings component tests)
registry: mgmt.yaml mgmt.router_settings.*
verify: open http://localhost:4000/ui/?page=router-settings, change a router setting, save, and check it via GET /config/field/info

### routing.max_parallel_requests: Concurrency controls (max_parallel_requests, admission queue)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/reliability, https://docs.litellm.ai/docs/proxy/high_throughput
code: `litellm/proxy/hooks/parallel_request_limiter.py`, `litellm/router.py` (`max_parallel_requests`)
tests: `tests/test_litellm/proxy/hooks/` (parallel request limiter files)
registry: none
verify: set `max_parallel_requests: 2` on a key, fire 5 concurrent calls, and confirm 3 are rejected or queued

### routing.rust_gateway: Rust gateway data plane
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/rust_gateway
code: `litellm-rust/`, `gateway/`
tests: `tests/e2e/llm_translation/test_ocr_rust_e2e.py` (rust-served endpoint coverage)
registry: none
verify: build and boot the rust gateway per the doc, then hit /v1/chat/completions through it and compare responses with the Python proxy
