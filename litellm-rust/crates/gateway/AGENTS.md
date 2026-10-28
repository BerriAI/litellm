- Keep this crate a thin composition layer: mount endpoint routers and serve the supplied listener
- Server lifecycle and shared inbound middleware belong here, including client authentication, rate limiting, and request logging
- Endpoint paths, request handling, model resolution, and response encoding belong to the mounted crates. Provider execution belongs to the `inference-<fmt>` crates and `llms`
- Inject shared state and infrastructure. Avoid global runtimes, duplicate client pools, and abstractions for hypothetical endpoint groups
- Test mounting and server lifecycle through public HTTP behavior. Test endpoint semantics in the owning crate

## Redis
- `build_redis` builds the Redis clients once at boot (`src/redis.rs`). Components get a clone as a field, never a per-request extractor or a pool. redis-rs multiplexes one connection per node, so there is no connection pool and pool settings like `max_connections` are accepted and ignored
- Which Redis backs which role mirrors the proxy: coordination is `general_settings.coordination_redis`, else the `cache_params` Redis when `cache: true` and `type: redis`, else `REDIS_*` env. The transaction buffer reuses coordination, else env when `use_redis_transaction_buffer` is on. The response cache gets its own client so large values never queue in front of coordination commands
- `litellm-config` follows `litellm/_redis.py`'s merge of config blocks and `REDIS_*` env. A redis-py setting we can't honor (sentinel, IAM, custom TLS files, ...) fails boot with its name, never silently dropped
- Boot never waits for a configured Redis (clients connect lazily). Only the env-only coordination fallback is pinged, for 2s, and treated as absent when unreachable, as in Python
- `/health/readiness` never touches Redis, matching the proxy
