# Redis

Background in the [Redis design](https://app.notion.com/p/Redis-40543b8acdab810b8ca6dbc39c3cf521). The native-parity gates and Rust token generation below supersede that page's earlier redis-py fallback and Python-token proposal

## Python surface today

- Cache callers (routing, auth cache, spend queues, cooldowns) go through `RedisCache` methods and `RedisBatch`, which apply the namespace and breaker
- Raw-access callers (rate limit, budget, lock, affinity and login Lua, config and auth-cache pub/sub, MCP OAuth) take clients and scripts from `RedisCache`. Raw clients skip the namespace and breaker, so callers namespace keys by hand
  - Scripts from `async_register_script` keep the namespace and breaker
- Vector search (`RedisSemanticCache`, `ValkeySemanticCache`, the Valkey vector store) builds its own clients
- `_redis.py` merges kwargs and `REDIS_*` env, picks auth and topology (standalone, cluster, Sentinel) and builds redis-py clients
- Keys, encoding, TTLs, Lua and reply decoding live in Python, spread across modules

## Shared crates

- `crates/config` resolves Redis settings for every host (`RedisSettings::resolve`)
- `crates/redis-client` owns connections, commands, pipelines, pub/sub, the namespace, the circuit breaker, TLS, Sentinel, cluster and cloud token providers. It knows nothing about domains
- `crates/redis-store` (planned) owns each keyspace's key layout, codec, expiry, Lua and reply decoding, like a `crates/db` repository
- Backend-independent cache identity stays out of stores, see [cache](cache.md)

## Per host

- python-sdk and python-proxy
  - Python passes the resolved kwargs, and the bridge feeds them to `RedisSettings::resolve`
  - Each migrated keyspace is a facade in `litellm/redis/stores/<keyspace>.py` over a `#[pyclass]` in `crates/python-bridge/src/redis/` that wraps the store. Transport and store pyclasses share one native client per resolved target
  - Cloud auth tokens are generated in Rust. A user-supplied `credential_provider` is the only Python token source and stays SDK-only
- rust-proxy
  - Builds the same settings from `config.yaml` and env, and calls `redis-store` directly

## Wire contracts

- What Redis sees stays exactly what Python writes today, during the migration and after it. This is permanent, not only until python-proxy is retired
- That covers key names (namespace and hash tags included), value bytes, TTLs, atomic units, and pub/sub channel names and payloads. The exact command sequence may differ when nothing above can tell
- So each keyspace keeps the shape Python uses today. A Lua keyspace stays one `EVAL`, a `DualCache` keyspace keeps its in-memory layer. Record the shape, atomic unit and who applies the namespace per keyspace

- Each move ships compatibility tests against one Redis. Old and new paths read each other's writes, and keys, values, expiry and atomicity hold through a rolling deploy

## Order and gates

1. `crates/redis-client` reaches parity with every config Python supports (standalone, cluster, Sentinel, TLS options, cloud auth, breaker)
2. Swap the transport. No redis-py fallback
3. Move keyspaces into `redis-store` one at a time. No Python fallback for a moved keyspace
4. Admin commands, pub/sub and vector search: TBD
5. Drop `redis-py` and `redisvl`, or keep them as optional extras: TBD
