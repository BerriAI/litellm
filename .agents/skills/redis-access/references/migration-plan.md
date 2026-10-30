# Redis migration plan

Two Python PRs first, then the Rust work. Milestones 1 and 2 are Python refactors that put every raw redis-py keyspace behind a store on the transport, so a keyspace can later move to Rust without touching its callers. They are not behavior free. Each lists its accepted behavior changes, and nothing outside that list may change. Milestones 3 and 4 are the Rust side and start only after milestone 2 merges, nothing in 1 or 2 depends on them or touches `litellm-rust/`

Every keyspace port after milestone 4 repeats its pattern, one keyspace at a time. `pr-sketches.md` has the interfaces each milestone brings

## 1. Transport, store layer and response polling (PR 1)

Python only

- `litellm/redis/transport/` with `RedisTransport`, `RedisPyTransport`, `RedisFailure`, `run_under_breaker` and `RedisServiceEvents`, built through `RedisCache.transport()`
- `run_under_breaker` wraps the existing `_run_under_circuit_breaker` in `litellm/caching/redis_cache.py`. Admission, the generation check, the swallowed-failure check, `_is_redis_health_failure` and the timeout flag stay exactly what `RedisCache` has today, the function only maps the outcome to a `RedisFailure`. It does not reimplement the breaker
- The transport takes the cache's `_admit_async_call`, so `RedisClusterCache`'s connection gate still applies to every command on the transport path
- Every "Change" item in `SKILL.md` applied before merge, plain methods instead of command dataclasses, `has_sinks()` checked before the events step walks the stack, `eval` taking script text with `EVALSHA` and `EVAL` fallback, tuples in replies
- `litellm/redis/stores/` with one keyspace moved, response polling, seeded by the old writer and read by the new reader and the reverse. One keyspace is enough to prove the shape, and it is the only one of the first slices that moves whole without a new Lua script
- Folds 48958, 48956, and from 49162 only `litellm/redis/stores/__init__.py` and the docs. The `litellm-rust/crates/redis-store` crate in 49162 waits for milestone 4. 49163 moved only `peek` of the MCP single-use marker, half a keyspace, so it folds into milestone 2 where the marker moves whole. 49164 (native transport) is closed, it mixes the two stacks

Accepted behavior changes

- REDIS service events carry the transport method name as `call_type` (`get`, `set`, `delete`, `eval`) instead of `async_get_cache` and friends. `litellm/integrations/prometheus_services.py` puts `call_type` into the `function_name` label, so that label's values change for moved keyspaces. Metric names and the other labels stay
- A stored value that does not decode is a `decode` failure in the store, never the `ast.literal_eval` fallback that `_get_cache_logic` has
- With no `service_callback` configured the events step neither walks the stack nor builds an event. The WARNING per failure still logs
- The REDIS service event is awaited in the caller's task instead of spawned as its own task
- The initial polling state is written through `json.dumps` like every later update, so its inner document gains the default `", "` and `": "` separators and ASCII escaping that `model_dump_json()` left out. Old and new readers both parse it, and every update already wrote that form

## 2. Every remaining raw redis-py keyspace behind a store (PR 2)

Python only. One commit per keyspace so each move reviews and reverts on its own. Split `parallel_request_limiter_v3` into its own PR if the diff gets large, it is the hot path

The list is the census of modules outside `litellm/caching/` that call `init_async_client()`, `.redis_client` or `async_register_script`, minus pub/sub and admin

- MCP single-use marker in `gateway_dcr_flow.py`, `peek` and `claim` together. `claim` becomes one `eval` doing `INCRBYFLOAT` plus `EXPIRE` when the key has no TTL, the bytes `async_increment` writes today
- Pod lock manager
- `parallel_request_limiter_v3` and its Lua. It also increments through `async_increment_cache_pipeline` on `DualCache`. Before splitting, check whether that pipeline touches the keys the scripts own. If it does, the keyspace moves whole and the transport grows what the pipeline does as one `eval`. If not, the pipeline stays on the generic verbs
- Max budget per session and max iterations limiters
- Batch enqueued tokens
- Login throttle
- Affinity cache
- MCP DCR client records and the outbound credential refresh coordinator

Accepted behavior changes

- Pod lock release is the compare-and-delete Lua only. The GET plus DEL fallback goes, since `eval` already falls back from `EVALSHA` to `EVAL`. Before merge, confirm no supported Redis deployment disables scripting. If one does, the fallback stays in `PodLockManager` as caller policy, never in the store
- Each caller's `call_type` and decode changes as in milestone 1

Done when nothing outside `litellm/redis/` and `litellm/caching/` calls `.redis_client`, `init_async_client` or `async_register_script` for these keyspaces, and a lint rule bans importing redis-py outside `litellm/redis/transport/redis_py.py`. The allowlist is `litellm/caching/redis_cache.py` and `litellm/caching/redis_cluster_cache.py`, which import `redis.exceptions` for the breaker and still own the generic verbs, plus the remaining users below. The list only shrinks

Stays on redis-py and out of this plan, per `SKILL.md`

- Pub/sub (`config_sync_pubsub`)
- Admin commands (`debug_utils`)
- Pipelines and `RedisBatch`
- Semantic and vector caches, which keep their own clients
- The generic `RedisCache.async_*` verbs used through `DualCache`, which move onto the transport one method at a time as separate work
- Spend update buffer and `batch_redis_get`. They never touch redis-py directly, they use `async_rpush_pipeline`, `async_lpop_pipeline`, `async_rpush_and_trim` and `async_batch_get_cache`, so they move with the generic verbs and pipelines, not here

## 3. Rust client parity

Rust only, nothing calls the new paths yet. This is 49189, on top of 49186 (SecretValue). It does not block or depend on 1 and 2

- Script ordering in pipelines, TTL in the same script for counter writes, breaker classification per topology matching redis-py, redis-py timeout and reconnect defaults
- TLS, Sentinel and cloud auth config, so fewer deployments need the transport-backed fallback in milestone 4

## 4. Bridge and the first native store

Starts after 2 merges, needs 3

- `litellm-rust/crates/redis-store` with the pod lock keyspace as async fns on `crates/redis-client`
- One shared native client handle in `crates/python-bridge/src/redis/` and one `#[pyclass]` for the lock store on it
- The breaker transition host call folding the Rust breaker into `redis_breaker_state_counts()`
- `NativePodLockStore` behind the existing `Protocol`, each operation through `RedisServiceEvents.observe`, picked at construction, with the transport-backed store kept for config the Rust client does not support yet
- Cross-implementation tests on real Redis from `litellm-redis-testing` (`litellm-rust/crates/redis-testing`), Python writes and Rust reads and the reverse, covering keys, values, expiry and owner-checked release

Pod lock goes first because it is low traffic and its owner-checked release only works if both stacks write the same `json.dumps` encoded owner

Switching the pod lock caller to the native store is its own change with live verification against a proxy, either a follow-up PR or a separately proven final commit
