# Redis stores

A store owns one keyspace. Read this before adding coordination state, moving a keyspace out of its caller, or porting a store to Rust

## Why stores

Keys, encoding, TTLs, Lua and reply decoding for each keyspace are spread across Python modules today, and Rust already has a second response-cache key builder in `cache_response::cache_key`. Python and Rust pods share one Redis, so if their keys or values differ they silently split counters, cooldowns or locks. Each shared keyspace needs one definition of its data and one set of operations, and the store is that definition

## Shape

- One module per keyspace in `litellm/redis/stores/<keyspace>.py`
- The module defines a `Protocol` of domain operations, named for what the caller wants (`acquire`, `release`, `increment`, `get`), never for Redis commands
- The Python implementation is a frozen dataclass that takes a `RedisTransport` and runs its own commands on it. It owns, and nothing outside it knows
  - the key builders, with logical keys and no namespace
  - the value codec
  - expiry, taken from the caller
  - Lua, once the transport runs it
  - reply decoding into typed results
- Every operation returns its result or a `RedisFailure`. A value that doesn't decode is a `decode` failure, never a miss or a default
- No logging, no retries, no fallback. Those are policy

```python
class SessionBudgetStore(Protocol):
    async def increment(self, session_id: str, cost: float, *, ttl: timedelta) -> float | RedisFailure: ...
    async def read(self, session_id: str) -> float | None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportSessionBudgetStore:
    transport: RedisTransport
    ...
```

## Callers

- Depend on the keyspace's `Protocol`, built once from `redis_cache.transport()` and injected where the caller can take it
- `match` the result, ending in `assert_never` when it has more than two cases
- Keep policy in the caller: which TTL applies, fail open or closed, and the `with_service_target` that labels its commands. The transport already logs failures and emits service events
- A caller never builds a key, touches a client or runs a command for a keyspace that has a store. Exposing a key builder for an existing public helper is fine as a delegate

## Adding new coordination state

Add a store, not a new `RedisCache` method. A Python store is the finished state for now, and adding one does not require a Rust port

## Moving an existing keyspace into a store

- Move the key builders, codec, TTLs, Lua and decoding verbatim. Keep policy in the caller
- Keys and values already in Redis are a wire format. Old and new pods read each other's writes during a rolling deploy, so a move keeps them byte for byte
- Changing a key, codec or TTL unit is never part of a move. It is its own change that writes a new versioned key and reads both for one release
- Move one keyspace per change

Keyspaces still in their callers, as candidates (paths under `litellm/`, not exhaustive)

- response cache, `caching/caching.py`
- rate limit windows, `proxy/hooks/parallel_request_limiter_v3.py`
- deployment cooldowns, `router_utils/cooldown_cache.py`
- routing affinity, `caching/affinity_cache.py`
- pod and MCP refresh locks, `proxy/db/db_transaction_queue/pod_lock_manager.py` and `proxy/mcp_server/outbound_credentials/redis_distributed_lock.py`, which share one owner-checked release
- login throttle, `proxy/auth/login_throttle.py`
- session caps, `proxy/hooks/max_budget_per_session_limiter.py` and `max_iterations_limiter.py`, which share one counter pattern
- enqueued batch tokens, `proxy/hooks/batch_enqueued_tokens.py`
- spend counters, `proxy/spend_tracking/spend_counter_batch.py`

Response cache identity (normalization, hashing into a logical key) is backend-independent and is not store logic. The store only maps that logical key to storage. Reuse or move `cache_response::cache_key`, never write a third key builder

## Tests

- Test the store through its caller against `fake_redis_cache(server)` from `tests/_support/fake_redis.py`. That is a real `RedisCache` on a `fakeredis.FakeServer`, so namespace, breaker and failure mapping are the production ones
- Cover each result the caller acts on, that the TTL lands (`pttl` on a raw `FakeAsyncRedis`), and an outage (`server.connected = False`)
- When a keyspace moves, seed with the old writer and read back with the old reader, so old and new pods read each other's writes
- Assert returned values and Redis state, never the commands sent

## Porting a store to Rust

Confirm with the user before starting the first port, since the bridge for it doesn't exist yet

- The keyspace's module in `litellm-rust/crates/redis-store` implements the same operations as async fns on `crates/redis-client`. Its `AGENTS.md` has the crate rules
- A `#[pyclass]` per store in `litellm-rust/crates/python-bridge/src/redis/` wraps that module, built with the shared `NativeRedisClient`, so Python opens no extra connections
- A `Native<Keyspace>Store` in the Python module implements the same `Protocol` over the pyclass, so callers don't change. The implementation is picked at construction, and the transport-backed one stays for config the native client doesn't support (Sentinel, cloud auth, TLS cert options)
- Results come back as Pydantic models generated from the Rust types, frozen, so callers keep matching on the same classes. A decode error maps to `RedisFailure(kind="decode")`
- Rust callers use the same module, so there is one implementation per operation
- Native stores skip the Python guard, so the breaker and the REDIS service events have to come from Rust before the first port
- The codec has to write the exact bytes Python writes. Pod lock owners are `json.dumps` output, which escapes non-ASCII, and `serde_json` doesn't, so a Rust pod couldn't release a lock a Python pod took
- Each port ships cross-implementation tests against one Redis, covering keys, values, expiry and atomicity through a rolling deploy
