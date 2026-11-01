# RedisTransport

How Python code reaches Redis. Read this before adding a command, or moving raw redis-py access off `RedisCache`

## Why it exists

Today commands reach Redis three ways

- Cache methods on `RedisCache` and `RedisBatch` (get, set, incr, lists, pipelines), which apply the namespace and the circuit breaker
- Raw access, where callers take `init_async_client()`, `init_pubsub_client()`, `async_register_script()` or `.redis_client` and run commands, Lua and pub/sub themselves
  - Scripts from `async_register_script` keep the namespace and breaker
  - Raw clients skip both, so every caller namespaces keys and channels by hand, and some forget
- Vector search (`RedisSemanticCache` through RedisVL, `ValkeySemanticCache` and the Valkey vector store through `Redis.from_url`), which builds its own clients on separate connections

`RedisTransport` replaces the raw path with one command-shaped seam, so the backend behind it can later swap from redis-py to `litellm-rust/crates/redis-client` without touching callers

## The contract

Defined in `litellm/redis/transport/contract.py`

- Commands are frozen dataclasses, `Get(key)`, `Set(key, value, ttl, nx=False)` and `Delete(key)`. `Arg` is `str | bytes | int | float`
- The transport has one method, `execute(command)`, overloaded per command for its reply type (`bytes | None`, `bool`, `int`)
- Every command returns its value or a `RedisFailure`. Nothing raises for a Redis problem
- `RedisFailure.kind`
  - `transport` is a timeout or lost connection. The command may still have applied, so a write that fails this way has an unknown outcome. `timeout=True` marks timeouts
  - `server` means Redis answered with an error (WRONGTYPE). The connection is fine and the breaker does not count it
  - `decode` means the reply was not the shape the method or store expected
  - `breaker_open` means the breaker refused the call without touching the socket
- The namespace is applied once, inside the transport. A key that already starts with `{namespace}:` is left alone, matching `RedisCache.check_and_fix_namespace` and `crates/redis-client`. Callers and stores pass logical keys
- Every write takes its TTL from the caller. `ttl=None` means no expiry, there is no default
- Replies are undecoded bytes. Decoding belongs to whoever owns the keyspace

These rules are shared with `crates/redis-client` (its `AGENTS.md` has the Rust side). If you change one, change both, since Python and Rust pods share keys in one Redis

## The redis-py backend

- `litellm/redis/transport/redis_py.py` holds `RedisPyTransport`, the only module that touches redis-py's client
- `RedisCache.transport()` builds one over that cache's own async client, namespace and circuit breaker. No new connection, no second breaker
- The guard is `BreakerGuard` in `redis_cache.py`. It runs the call under the cache's breaker and maps exceptions with `_redis_failure`
  - `ResponseError` / `DataError` that are not health failures become `server`
  - health failures and other `RedisError`s become `transport`
  - anything else re-raises, since it is a bug, not a Redis failure
- `BreakerGuard` wraps `ReportingGuard`, which owns observability for every command
  - one REDIS service success or failure event per command, with the command name as `call_type`, sent to the cache's `service_logger_obj`. A `breaker_open` refusal sends none
  - the event and the log line carry the caller's `with_service_target`, so callers label their commands and emit nothing themselves
  - one WARNING per failure through the timeout log throttle, `breaker_open` at DEBUG
- Callers don't log Redis health. Log only what a failure means for the feature, and only when it adds something the transport line doesn't say

## Adding a command

- Add a frozen dataclass for the command to `contract.py`, add it to `Command`, and add an `execute` overload for its reply type. Keep it command-shaped, never redis-py shaped (no `**kwargs`, no client options), so the native client fits behind it unchanged
- Add a `match` arm in every backend's `execute`, and add the redis-py method to the `AsyncRedisCommands` Protocol there and `_AsyncRedisCommands` in `redis_cache.py`
- Namespace every key argument with `_key`, take TTLs as `timedelta`, send them as milliseconds
- Validate the reply shape and return `decode` for anything unexpected
- Test it in `tests/unit/redis/transport/test_redis_py.py` against fakeredis: the namespace is applied once, TTL lands, a rejected command is `server`, an outage is `transport`. Service events and logging come from the guard, so a new command needs no test of its own for them
- Multi-key commands must keep their keys in one slot, or they break on cluster depending on slot ownership

## Moving raw access onto the transport

- Replace the raw client or script handle with transport calls, and delete hand-written namespacing and script registration
- Keep the caller's outcome per failure. If it failed closed on any exception, it fails closed on any `RedisFailure`
- Don't treat a `transport` failure on a write as "not applied"

## Not on the transport yet

- Pub/sub (config and auth-cache invalidation), admin commands, pipelines and `RedisBatch` stay on `RedisCache` until the transport grows them
- Vector and semantic search keep their own clients. The plan gives them native index handles on `crates/redis-client`, not `RedisTransport`

## Where it's going

- A `NativeRedisTransport` (a PyO3 adapter over one `RedisClient` per resolved target) goes behind the same Protocol and takes the same `Guard`. The backend is picked at construction, with redis-py kept for config the native client doesn't support yet (Sentinel, cloud auth, TLS cert options)
- Settings for both stacks resolve through `crates/config` (`RedisSettings.resolve`), so Python and Rust connect to the same target with the same namespace
- During the swap the Python `RedisCircuitBreaker` still wraps the transport. The breaker and the service events have to be in Rust before any store runs natively, because native stores call `crates/redis-client` directly
- Once stores and admin/pub-sub have replacements, redis-py and redisvl are dropped or become optional extras

Don't build any of this unless asked
