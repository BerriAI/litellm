- Keep this crate a thin composition layer: mount endpoint routers and serve the supplied listener
- Server lifecycle and shared inbound middleware belong here, including client authentication, rate limiting, and request logging
- Inference routes authenticate with `BearerOrXApiKey`, since the proxy accepts an Anthropic `x-api-key` there
- Endpoint paths, request handling, model resolution, and response encoding belong to the mounted crates. Provider execution belongs to the `inference-<fmt>` crates and `llms`
- Inject shared state and infrastructure. Avoid global runtimes, duplicate client pools, and abstractions for hypothetical endpoint groups
- Test mounting and server lifecycle through public HTTP behavior. Test endpoint semantics in the owning crate
- Standalone and in front of Python serve the same native app, built by `native` in `src/main.rs`. In front of Python the only addition is `admit` (`src/admit/mod.rs`) around all of it, so never give one mode a layer the other lacks
- `admit` is the only thing that sends a request to Python. It sees each request before any layer, and every method and path it admits must have a native route, because a request it admits that no route takes gets the native 404 and is never forwarded to Python
- What `admit` leaves to Python is decided by the rules under Admit below

## Admit
- A request is admitted only when nothing it touches is unjudged. Every other request goes to Python as the client sent it, with why in `x-litellm-gateway`
- The decision is pure. `survey.rs` turns the config into findings, `plan.rs` holds them and answers for one request, and `request.rs` finds what a request carries. None of them does IO or reads the environment. `mod.rs` is the only part that does: it looks up what the config does not say (the router's deployments, whether there is a database, which environment switches are set), checks the caller and reads the body
- A finding is one setting the native app does not honor as Python does, with the scope it leaves to Python
  - `Everything`, for a setting that reaches every request, as a callback or `custom_auth` does
  - `Group`, for a deployment's own setting or provider. It leaves that model group and every group that can fall back to it, and no other group
  - `Asking`, for a setting that only matters to a request carrying a trait, as an unset `drop_params` only matters to a coding agent
- Each config section in `survey.rs` is destructured field by field, so a new config field does not compile until it is honored (`field: _`) or named as a finding. A key the config types do not name is always a finding. Never add a catch-all `..`
- A setting is honored when the native app reads it as Python does, or when it only shapes what `admit` never admits: another route, or a caller that holds neither the master key nor a virtual key
- Rules exist only for what the native app lacks. Porting a feature deletes its finding or its trait in the same change, and finding a new gap adds one. Never keep a rule for something the native app already does as Python does
- Forward only what is opt-in and can be told from the config or the request before any side effect. What sits on the default path has to be ported, and its finding is the list of what still blocks a config from being served natively
- `MESSAGES_PROVIDERS` in `survey.rs` is the list of providers the native `/v1/messages` has been compared with Python on. Add a provider only with a live comparison
- A trait in `request.rs` is found from the request alone, and `Trait::is_read` says whether the native app reads it as Python does unless a finding names it. A trait the native app never reads needs no finding
- A request trait that is admitted only because a setting is refused is tied to that setting. Session headers are admitted because no callback or agent reads them and the native app records no spend, so the change that makes callbacks or spend native has to emit the traits it does not handle yet
- A model the config does not name is left to Python, which also resolves wildcards, deployment ids and aliases the gateway does not
- Checks run cheapest first: the settings that leave everything, the route, the caller, the headers, then the body. The body is read only when every earlier check passed
- The body is parsed once. `admit` judges the parsed object by its keys and hands it to the route as the `JsonObject` extension, so the route does not parse it again. Only an admitted request carries it. Whether each value has the right type is left to the route
- Boot logs every finding with its setting and its scope, and each refusal logs what it was over at debug. The header keeps the short reason, never a setting name
- A database no longer leaves a request to Python. The caller check reads it for a virtual key (see Keys), and a config or `STORE_MODEL_IN_DB` that turns `store_model_in_db` on still leaves everything
- Not ported yet, and not a finding either: what Python writes to its database after a call. A request the native app answers gets no spend log row and adds nothing to its key's spend
- Not judged yet: models and settings Python holds in its database, which it also reads when only the database turns `store_model_in_db` on, CLI flags Python was started with, a Redis only `REDIS_*` configures, and environment variables outside `Switch`

## Keys
- With a database, `AppState::auth` authenticates the master key and a virtual key, in both modes. `StoredKeys` (`src/keys.rs`) reads the key's row from Postgres on every request, with no cache in front of it
- The native app honors a key's expiry and its block and nothing else. `carried` takes the row apart column by column, and a key that holds anything else (a team, an owner, a budget, a limit, a model list, metadata) is `Forbidden`, which `admit` leaves to Python as `credential`. A column added to the row does not compile until it is honored or named
- Honoring a column natively moves it out of `carried` in the same change. Never report a key active while a column it holds is not enforced
- `admit` and the route's `authenticate` layer each check the caller, so an admitted virtual key costs two reads

## Resources
- `Resources::build` (`src/resources.rs`) makes what every part is built from, once per config: the environment (the config's `environment_variables` in front of the process environment), the `SecretResolver` over it, the HTTP client config and the inference `CoreResources` with the client pool. Inference, Redis and the database each take what they need from it. Never reach into one part for another part's pool, config or secrets, and never build a second environment overlay
- A setting that can come from the environment, the secret manager or OIDC is read through `resources.secrets`. `resources.environment` is only for what Python reads straight from `os.environ`

## State
- `AppState::build` (`src/state.rs`) builds `Resources` and hands them to `AppState::over`, which builds what both modes share once at boot: inference, the Redis clients, the database pools and the key lookup over them. `router` takes it and hands it to routes with `with_state`, and a handler or middleware takes the part it needs with `State<RedisClients>` or `State<Option<Pools>>`, which `FromRef` clones out of it. Never build a store inside a route and never give one mode a store the other lacks
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
- `db::connect` (`src/db.rs`) builds `litellm_db::Pools`, and the `litellm_repositories::Database` the repositories run their queries on, from `resolve` in `db-resolve`, which takes `general_settings` and `resources.secrets` the way `redis-resolve`'s `resolve` does. No database configured, or a token auth toggle Rust cannot mint for yet, is `None`
- The pools are lazy, so boot never waits for Postgres. A handler takes `State<Option<Pools>>` and picks `writer()`, `read()` or `read_after_write()` per query. Don't add an extractor that checks out a connection, because it would hide that choice
- `/health/readiness` pings the writer for up to 2s and reports `connected`, `disconnected`, or `Not connected` when no database is configured, and always answers 200. The proxy's 503 on `disconnected` without `allow_requests_on_db_unavailable` and its 15s cache of a good answer are not ported

## License
- `AppState::build` verifies the license once at boot with `build_license` (`src/license.rs`) and keeps the result as `entitlements`, which never changes while serving. `middleware::global` hands it to every middleware and route as `Extension<Arc<Entitlements>>`
- The key comes from `LITELLM_LICENSE`, read through `resources.environment` so config `environment_variables` replaces the process environment, and from `general_settings.litellm_license`, which wins unless only the environment's key is enterprise, as in `_use_config_license`. The license server is asked through the `resources` pool and HTTP config
- Verification itself belongs to `litellm-license`. A key the bundled public key did not sign is asked of the license server, so boot can wait on that request for `NON_LLM_CONNECTION_TIMEOUT` seconds per attempt
- A setting Python's router honors and the native host does not yet is refused, never ignored: `boot::slot_limit` names a config that asks for `max_parallel_requests` slots, and the build fails on it
