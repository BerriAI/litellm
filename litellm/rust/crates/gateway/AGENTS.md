- Keep this crate a thin composition layer: mount endpoint routers and serve the supplied listener
- Server lifecycle and shared inbound middleware belong here, including client authentication, rate limiting, and request logging
- Inference routes authenticate with `BearerOrXApiKey`, since the proxy accepts an Anthropic `x-api-key` there
- Endpoint paths, request handling, model resolution, and response encoding belong to the mounted crates. Provider execution belongs to the `inference-<fmt>` crates and `llms`
- Inject shared state and infrastructure. Avoid global runtimes, duplicate client pools, and abstractions for hypothetical endpoint groups
- Test mounting and server lifecycle through public HTTP behavior. Test endpoint semantics in the owning crate

## Resources
- `Resources::build` (`src/resources.rs`) makes what every part is built from, once per config: the environment (the config's `environment_variables` in front of the process environment), the `SecretResolver` over it, the HTTP client config and the inference `CoreResources` with the client pool. Inference, Redis and the database each take what they need from it. Never reach into one part for another part's pool, config or secrets, and never build a second environment overlay
- A setting that can come from the environment, the secret manager or OIDC is read through `resources.secrets`. `resources.environment` is only for what Python reads straight from `os.environ`

## State
- `AppState::build` (`src/state.rs`) builds `Resources` and then the stores once at boot: inference, the Redis clients and the database pools. `router` takes it and hands it to routes with `with_state`, and a handler or middleware takes the part it needs with `State<RedisClients>` or `State<Option<Pools>>`, which `FromRef` clones out of it. Never build a store inside a route
- Forwarding mode in `src/main.rs` serves nothing natively, so it builds no state. Once it serves a native app it calls the same `AppState::build`, so the two modes never differ in their stores

## Redis
- `redis::connect` builds the Redis clients once at boot (`src/redis.rs`). `State<RedisClients>` hands out clones of them, never a connection checked out per request. redis-rs multiplexes one connection per node, so there is no connection pool and pool settings like `max_connections` are accepted and ignored
- Which Redis backs which role mirrors the proxy: coordination is `general_settings.coordination_redis`, else the `cache_params` Redis when `cache: true` and `type: redis`, else `REDIS_*` env. The transaction buffer reuses coordination, else env when `use_redis_transaction_buffer` is on. The response cache gets its own client so large values never queue in front of coordination commands
- Merging a configured block with `REDIS_*` and the `get_secret` fallbacks is `resolve` in `redis-resolve`, which only this host calls because the Python host merges in Python. `src/redis.rs` only picks the `Surface` (`Coordination`, `CacheParams` or `Environment`) and calls it with `resources.secrets`
- A setting the Rust client cannot honor warns and is dropped, and only a security check it cannot enforce fails boot, per `redis-resolve/AGENTS.md`
- Boot never waits for a configured Redis (clients connect lazily). Only the env-only coordination fallback is pinged, for 2s, and treated as absent when unreachable, as in Python
- `/health/readiness` never touches Redis, matching the proxy

## Database
- `db::connect` (`src/db.rs`) builds `litellm_db::Pools` from `resolve` in `db-resolve`, which takes `general_settings` and `resources.secrets` the way `redis-resolve`'s `resolve` does. No database configured, or a token auth toggle Rust cannot mint for yet, is `None`
- The pools are lazy, so boot never waits for Postgres. A handler takes `State<Option<Pools>>` and picks `writer()`, `read()` or `read_after_write()` per query. Don't add an extractor that checks out a connection, because it would hide that choice
- `/health/readiness` pings the writer for up to 2s and reports `connected`, `disconnected`, or `Not connected` when no database is configured, and always answers 200. The proxy's 503 on `disconnected` without `allow_requests_on_db_unavailable` and its 15s cache of a good answer are not ported

## License
- `build_license` verifies the license once at boot (`src/license.rs`) and the result never changes while serving. For now the standalone gateway only logs whether it is enterprise
- The key comes from `LITELLM_LICENSE`, where config `environment_variables` replaces the process environment, and from `general_settings.litellm_license`, which wins unless only the environment's key is enterprise, as in `_use_config_license`
- Verification itself belongs to `litellm-license`. A key the bundled public key did not sign is asked of the license server, so boot can wait on that request for `NON_LLM_CONNECTION_TIMEOUT` seconds per attempt
