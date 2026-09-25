# 07 Caching and cost optimization

Response caches that answer repeated prompts without an upstream call, cache controls, and the cost-cutting features (prompt compression, the cost optimization page). Prompt-cache-aware routing decisions live in [03-routing-reliability.md](03-routing-reliability.md) (`routing.sticky_sessions`), and auto-router savings overlap with `routing.auto_router` in the same file

## Response caching

### cache.response_cache: Response caching (in-memory, redis, redis cluster, s3/gcs, disk, qdrant)
surfaces: sdk, config, ui | flags: redis
docs: https://docs.litellm.ai/docs/proxy/caching, https://docs.litellm.ai/docs/caching/all_caches, https://docs.litellm.ai/docs/proxy/caching_redis, https://docs.litellm.ai/docs/proxy/caching_object_storage
code: `litellm/caching/`, `litellm/caching/caching.py`, `litellm/caching/base_cache.py`, `litellm/caching/caching_handler.py`
tests: `tests/test_litellm/caching/`, `tests/e2e/router/` (cache behavior), `tests/test_litellm/caching/test_embedding_router.py`
registry: reliability.yaml reliability.cache.*
verify: set `litellm_settings.cache: true` with redis params, send the same chat completion twice, and confirm the second returns instantly with `x-litellm-cache-key` semantics (cache hit)

### cache.semantic: Semantic caching
surfaces: sdk, config | flags: redis
docs: https://docs.litellm.ai/docs/proxy/caching_semantic
code: `litellm/caching/`, `litellm/caching/caching.py`
tests: `tests/test_litellm/caching/`
registry: none
verify: enable semantic caching per the doc, send two differently-worded but equivalent prompts, and confirm the second is served from cache

### cache.controls: Cache controls (ttl, no-cache, no-store, namespace, per-key/team)
surfaces: sdk, api, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/caching_controls, https://docs.litellm.ai/docs/proxy/caching_settings
code: `litellm/caching/caching.py` (ttl, namespace), `litellm/proxy/management_endpoints/cache_settings_endpoints.py` (`/cache/settings`), `litellm/proxy/litellm_pre_call_utils.py` (per-request cache flags)
tests: `tests/test_litellm/caching/`
registry: none
verify: POST /cache/settings with a ttl, send a call with `cache: {"no-store": true}` in the body, and confirm it bypasses the cache

### cache.cloud_auth: Redis auth (ElastiCache IAM, Memorystore IAM, Azure AD)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/elasticache_iam, https://docs.litellm.ai/docs/proxy/gcp_memorystore_iam, https://docs.litellm.ai/docs/proxy/azure_redis_ad
code: `litellm/_redis.py` if present, `litellm/caching/` (redis client init)
tests: `tests/test_litellm/caching/` (redis auth files)
registry: none
verify: configure redis_cache with the IAM auth block for your cloud, boot, and confirm cache reads/writes work without a password

### cache.ui: Response Cache page and cache settings
surfaces: ui, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/caching_settings
code: `litellm/proxy/management_endpoints/cache_settings_endpoints.py` (GET/POST `/cache/settings`, `/cache/settings/test`), `ui/litellm-dashboard/src/app/(dashboard)/caching/page.tsx`
tests: `tests/test_litellm/proxy/management_endpoints/` (cache settings files)
registry: none
verify: open http://localhost:4000/ui/?page=caching, set a cache config, hit /cache/settings/test, and confirm the connection check passes

## Cost optimization

### costopt.prompt_compression: Prompt compression (compress(), compresr, headroom)
surfaces: sdk, api, config | flags: none
docs: https://docs.litellm.ai/docs/completion/prompt_compression, https://docs.litellm.ai/docs/proxy/guardrails/compresr, https://docs.litellm.ai/docs/proxy/headroom
code: `litellm/compression/compress.py`, `litellm/integrations/compression_interception/`, `litellm/proxy/guardrails/guardrail_hooks/compresr/`, `litellm/proxy/guardrails/guardrail_hooks/headroom/`, `litellm/proxy/spend_tracking/compression_savings.py`
tests: `tests/test_litellm/test_compression.py`, `tests/test_litellm/proxy/spend_tracking/test_compression_savings.py`
registry: guardrail.yaml guardrail.headroom.*
verify: enable a compression hook in config, send a long prompt, and compare usage.prompt_tokens against the uncompressed call

### costopt.page: Cost optimization page
surfaces: ui, api | flags: db
docs: none found
code: `ui/litellm-dashboard/src/app/(dashboard)/cost-optimization/page.tsx`, `litellm/proxy/spend_tracking/compression_savings.py`, `litellm/proxy/spend_tracking/spend_management_endpoints.py`
tests: `tests/test_litellm/proxy/spend_tracking/`
registry: none
verify: run compressed and cached calls, open http://localhost:4000/ui/?page=cost-optimization, and confirm savings numbers render

### costopt.auto_router_savings: Auto-router cost savings
surfaces: ui, api | flags: none
docs: https://docs.litellm.ai/docs/auto_router/benchmarks, https://docs.litellm.ai/docs/tutorials/claude_code_cut_costs
code: `litellm/proxy/management_endpoints/auto_router_endpoints.py` (`/auto_router/benchmarks`, `/auto_router/shadow_eval*`), `litellm/proxy/guardrails/auto_router_compression.py`
tests: `tests/test_litellm/proxy/guardrails/test_auto_router_compression.py`
registry: reliability.yaml reliability.routing.*
verify: run /auto_router/shadow_eval over a traffic window and read the projected savings; see also `routing.auto_router` in [03-routing-reliability.md](03-routing-reliability.md)

### costopt.prompt_cache_routing: Prompt cache aware routing
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/auto_router/prompt_caching, https://docs.litellm.ai/docs/tutorials/claude_code_prompt_cache_routing
code: `litellm/router_utils/prompt_caching_cache.py`, `litellm/proxy/hooks/prompt_cache_prediction.py`
tests: `tests/test_litellm/router_utils/`
registry: none
verify: enable prompt-cache-aware routing, send a long repeated-prefix conversation twice, and confirm it pins to the warm deployment (see `routing.sticky_sessions`)
