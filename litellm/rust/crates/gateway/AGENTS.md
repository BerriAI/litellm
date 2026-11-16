- Keep this crate a thin composition layer: mount endpoint routers and serve the supplied listener
- Server lifecycle and shared inbound middleware belong here, including client authentication, rate limiting, and request logging
- Inference routes authenticate with `BearerOrXApiKey`, since the proxy accepts an Anthropic `x-api-key` there
- Endpoint paths, request handling, model resolution, and response encoding belong to the mounted crates. Provider execution belongs to the `inference-<fmt>` crates and `llms`
- Inject shared state and infrastructure. Avoid global runtimes, duplicate client pools, and abstractions for hypothetical endpoint groups
- Test mounting and server lifecycle through public HTTP behavior. Test endpoint semantics in the owning crate
- Standalone and in front of Python serve the same native app, built by `native` in `src/main.rs`. In front of Python the only addition is `admit` (`src/admit/mod.rs`) around all of it, so never give one mode a layer the other lacks
- `admit` is the only thing that sends a request to Python. It sees each request before any layer, and every method and path it admits must have a native route, because a request it admits that no route takes gets the native 404 and is never forwarded to Python
- `Admit` is built once from the config the native app is built from, and judges each request by its route and its caller, where only the master key is checked. A request that passes is answered natively. It does not judge the config's settings or the request's headers and body yet, so a setting or parameter only Python honors is served natively. Those judges, and a config Python can reload from its database, come in a later change, and forwarding must not be turned on for users before it

## Resources
- `Resources::build` (`src/resources.rs`) makes what every part is built from, once per config: the environment (the config's `environment_variables` in front of the process environment), the `SecretResolver` over it, the HTTP client config and the inference `CoreResources` with the client pool. Inference, Redis and the database each take what they need from it. Never reach into one part for another part's pool, config or secrets, and never build a second environment overlay
- A setting that can come from the environment, the secret manager or OIDC is read through `resources.secrets`. `resources.environment` is only for what Python reads straight from `os.environ`

## State
- `AppState::build` (`src/state.rs`) builds `Resources` and then what both modes share once at boot: inference, the Redis clients and the database pools. `router` takes it and hands it to routes with `with_state`, and a handler or middleware takes the part it needs with `State<RedisClients>` or `State<Option<Pools>>`, which `FromRef` clones out of it. Never build a store inside a route and never give one mode a store the other lacks
- In front of Python the config is the local YAML file Python loads, which the supervisor names in `LITELLM_FORWARD_CONFIG`. When Python loads no local file, `load_config` cannot load the one it names, or the native app cannot be built from it, `native_state` in `src/main.rs` builds the state from `Config::default()`, so only the environment's `REDIS_*` and `DATABASE_*` are seen, and `Admit::unconfigured` leaves every request to Python. Never fail boot on that file or on what it sets, because the gateway exiting stops Python too. Config stored in the database is not read yet

## Host
- `src/host/` is the native host, the counterpart of python-bridge's `PythonRouterHost`. `NativeBootHost` answers router-core's deployment build at boot, and `NativeHost` builds a `NativeRouterHost` for each inference call. `Gateway` is gateway-inference's `Gateway<NativeHost>`, so the routes are served through it without depending on it
- A hostcall answered from the gateway's own state belongs here, not in gateway-inference. Only running the route's attempt and authorizing a name stay with the routes

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
- `AppState::build` verifies the license once at boot with `build_license` (`src/license.rs`) and keeps the result as `entitlements`, which never changes while serving. `middleware::global` hands it to every middleware and route as `Extension<Arc<Entitlements>>`
- The key comes from `LITELLM_LICENSE`, read through `resources.environment` so config `environment_variables` replaces the process environment, and from `general_settings.litellm_license`, which wins unless only the environment's key is enterprise, as in `_use_config_license`. The license server is asked through the `resources` pool and HTTP config
- Verification itself belongs to `litellm-license`. A key the bundled public key did not sign is asked of the license server, so boot can wait on that request for `NON_LLM_CONNECTION_TIMEOUT` seconds per attempt
- A setting Python's router honors and the native host does not yet is refused, never ignored: `boot::slot_limit` names a config that asks for `max_parallel_requests` slots, and the build fails on it
