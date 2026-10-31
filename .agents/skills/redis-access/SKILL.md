---
name: redis-access
description: Add Redis operations, move raw redis-py access or a keyspace behind RedisTransport and a store in litellm/redis, or explicitly port a store to Rust, without mixing the Python and Rust Redis stacks or expanding migration scope
metadata:
  internal: true
---

# Redis access

## Two stacks, never mixed

- Python keyspace code reaches Redis one way only, `Python store -> RedisPyTransport -> redis-py`
- Rust keyspace code reaches Redis one way only, `Python caller -> pyo3 -> crates/redis-store -> crates/redis-client`, and the Rust gateway enters at `crates/redis-store`
- A Python store never runs on the Rust client, and a Rust store never goes back through the Python transport. There is no Rust-backed `RedisTransport`, since that would be a second redis client binding over pyo3 that has to grow toward redis-py's whole command set
- A keyspace is ported as a unit, keys plus codec plus Lua plus client, or not at all. Which stack a keyspace is on is decided per keyspace, never per call
- During the transition a pod holds one redis-py pool and one Rust client to the same Redis, with one breaker each. That is correct, they guard different sockets. redis-py goes away only when the last keyspace, pub/sub and admin have a Rust home

## Adding new Redis access

- New coordination state gets a store in `litellm/redis/stores/<keyspace>.py`, not a new `RedisCache` method
- A Python store is a finished state. Adding one does not require a Rust port
- Callers get typed domain results from the store and keep policy, which TTL applies, fail open or closed, retries, and the `with_service_target` that labels the calls
- New code never calls `init_async_client`, `init_pubsub_client`, `async_register_script` or `.redis_client`, and never imports redis-py outside `litellm/redis/transport/redis_py.py`. `litellm/caching/redis_cache.py` and `redis_cluster_cache.py` stay allowlisted while they own the breaker's exception types and the generic verbs

## The transport

`RedisTransport` in `litellm/redis/transport/contract.py` is how Python code reaches redis-py. It is a Protocol of plain methods, command-shaped and never redis-py shaped, and it covers every Redis use in Python, so a keyspace move never waits on the transport

- Data commands `get`, `mget`, `set`, `delete`, `incrby`, `incrbyfloat`, `expire`, `ttl`, `rpush`, `lpop`, `ltrim`, `sadd`, `scan` and `eval`
- Batches, `batch()` for one caller's commands, `batch(transaction=True)` for MULTI/EXEC, and `request_batch()` / `post_call_batch()` for the request scope
- Pub/sub, `publish`, `publish_all` and `subscribe`
- Admin and health, `ping`, `info`, `client_list` and `flushall`

- Every method returns its value or a `RedisFailure`. Nothing raises for a Redis problem
- `RedisFailure.kind` is `transport` (timeout or lost connection, the command may still have applied, so a failed write is an unknown outcome, never "not applied"), `server` (Redis answered with an error, the connection is fine and the breaker does not count it), `decode` (the reply was not the shape expected) or `breaker_open` (refused without touching the socket)
- The namespace is applied once, inside the transport, to every key including Lua `KEYS` and pub/sub channels. A key that already starts with `{namespace}:` is left alone. Stores and callers pass logical keys
- Every write takes its TTL from the caller as a `timedelta`. `None` means no expiry, there is no default
- Replies are undecoded bytes. `eval` returns `int | bytes | None | tuple[...]`. Decoding belongs to the store that owns the keyspace
- `eval` takes script text every call and does `EVALSHA` with `EVAL` fallback, so nothing registers scripts
- These rules are shared with `crates/redis-client`, whose `AGENTS.md` has the Rust side. Change both together, since Python and Rust pods share keys in one Redis
- `RedisCache.transport()` builds the transport over that cache's own client, pub/sub client, namespace, circuit breaker, `_admit_async_call`, cluster-aware `mget` and `ServiceLogging`. No new connection, no second breaker, and `RedisClusterCache`'s connection gate still applies
- Every method runs its command through two steps, in this order. `run_under_breaker` wraps the existing `_run_under_circuit_breaker` from `litellm/caching/redis_cache.py`, so admission, the generation check, the health failure classification and the timeout flag are the ones legacy traffic uses, and maps the outcome to a `RedisFailure` (`breaker_open`, `transport`, `server`) instead of raising. `RedisServiceEvents.observe` times the call, logs a failure and emits the service event

- `ping`, the admin commands and pub/sub skip the breaker, like the `RedisCache` methods they replace, so health checks still reach a Redis the breaker took out. They still go through `RedisServiceEvents.observe`, except `get_message` polling

```python
@dataclass(frozen=True, slots=True)
class RedisPyTransport:
    client: Callable[[], AsyncRedisCommands]
    namespace: str | None
    breaker: RedisCircuitBreaker
    admit: Admit
    events: RedisServiceEvents
    ...

    async def get(self, key: str) -> bytes | None | RedisFailure:
        return _bytes_reply(await self._run("get", lambda: self.client().get(self._key(key))))

    async def _run(self, name: str, call: Callable[[], Awaitable[_T]]) -> _T | RedisFailure:
        return await self.events.observe(name, lambda: run_under_breaker(self.breaker, name, self.admit, call))
```
- A Redis command Python starts using is a method on the Protocol and on the batch, an implementation in `redis_py.py`, and a fakeredis test that the namespace is applied once, the TTL lands, a rejected command is `server` and an outage is `transport`
- A store still picks the fewest round trips. A counter with a TTL on first write or an owner-checked release is one `eval`, not three commands
- Multi-key commands and scripts keep their keys in one slot, or they break on cluster depending on slot ownership

## Batches

The pipeline engine is `PipelineBatch` in `litellm/redis/transport/batch.py`, the one `RedisBatch` has always had. Transport batches and `RedisBatch` (`litellm/caching/redis_batch.py`) declare onto it, so a request keeps one round trip per backend while callers move over one at a time

- Every batch method returns an awaitable reply, and awaiting any reply flushes everything declared so far. Each reply is its value or a `RedisFailure`, so one rejected command fails only its own reply, and an outage fails every reply and counts once on the breaker
- `request_batch()` and `post_call_batch()` return the engine `request_redis_batch_scope` holds for this backend, or `None` outside a request. Never build a second request scope
- A script Redis has not cached reruns on its own after the pipeline, so do not rely on its effect being visible to later commands in the same batch. A transaction batch sends `EVAL` instead, so its scripts run inside the MULTI
- On a cluster client a request batch runs each command on its own, as `RedisBatch` always has. Explicit and transaction batches still pipeline
- One REDIS event per flush, named by the batch and targeted by the key families that shared it, `mixed` with the sorted families when there are several

## Stores

A store owns one keyspace. Python and Rust pods share one Redis, so each keyspace needs one definition of its keys and values, and the store is that definition

- One module per keyspace. It defines a `Protocol` of domain operations named for what the caller wants (`acquire`, `release`, `increment`, `get`), never for Redis commands, and a frozen dataclass implementation that takes a `RedisTransport`
- The store owns, and nothing outside it knows, the key builders (logical keys, no namespace), the value codec, expiry taken from the caller, the Lua, and reply decoding into typed results
- Every operation returns its result or a `RedisFailure`. A value that does not decode is a `decode` failure, never a miss or a default
- No logging, no retries, no fallback in a store. Those are policy and stay in the caller
- A caller never builds a key, touches a client or runs a command for a keyspace that has a store. Exposing a key builder as a delegate for an existing public helper, or for the caller's in-memory fallback, is fine
- Store operations take domain inputs, the ids and typed values the caller already has, and never a Redis key, a bundle of keys or encoded bytes. A native store implements the same `Protocol`, so anything the caller hands over ready-made stays in Python after a Rust port
  - The store encodes what it writes and decodes what it reads, both ways. A caller never serializes a value before handing it over, and never parses a value a store returned
  - A caller never calls a store's key builder to pass the result back into the store

```python
# a caller building the key or the value
await store.claim(f"mcp_gateway_dcr_code_used:{jti}", ttl=ttl)
await store.acquire(refresh_lock_key(user_id, server_id), token, ttl=ttl)
await store.save_record(batch_id, adapter.dump_json(reservation).decode(), ttl=ttl)

# the store owning both
await store.claim(UsedCode(jti), ttl=ttl)
await store.acquire(user_id, server_id, token, ttl=ttl)
await store.save_record(batch_id, reservation, ttl=ttl)
```
- Keys and values already in Redis are a wire format. Old and new pods read each other's writes during a rolling deploy, so a move keeps them byte for byte, including `json.dumps` spacing and ASCII escaping, float formatting, and seconds versus milliseconds. Changing a key, codec or TTL unit is never part of a move. It is its own change that writes a new versioned key and reads both for one release
- Response cache identity (normalization, hashing into a logical key) is backend independent and is not store logic. Reuse or move the Rust `cache_response::cache_key`, never write a third key builder

## Asked to move existing access behind a store

- Move the key builders, codec, TTLs, Lua and decoding verbatim. Keep policy in the caller
- Keep the caller's outcome per failure. If it failed closed on any exception, it fails closed on any `RedisFailure`
  - Read how the replaced call failed before mapping it. `async_get_cache` and `async_set_cache` swallow Redis errors and return `None`, `async_delete_cache`, `async_increment` and registered scripts raise, and every method behind `_redis_circuit_breaker_guard` raises `RedisCircuitBreakerOpenError` when the breaker is open. So one `RedisFailure` can map to different outcomes by `kind`, and the caller matches on `kind` to keep each one
  - The one accepted difference is the breaker. A path that called the raw redis-py client skipped it and now goes through it. Say so in the PR
- The move leaves no caller holding a key or a codec for that keyspace. Change the operations to take domain inputs in the same change, never as a follow-up
- Delete the caller's own timing, service hooks and Redis health logging. The transport emits for it now
- Move one keyspace per change. An extraction is not a Rust port
- `RedisCache` is a connection object wearing a cache name. Its own cache verbs move onto the transport one method at a time, each conversion deleting that method's inline logging, until what is left is the connection, the breaker and a thin facade
- Pipelined and batched access moves onto a transport batch, pub/sub onto `publish` / `subscribe`, and admin onto the admin methods. Vector and semantic search keep their own clients

## Observability

`ServiceLogging` (`litellm/_service_logger.py`) is a shared infra-health channel with many producers (Redis, Postgres, the spend queues, auth, router) and three sinks (`prometheus_system`, `datadog`, OTel). Redis work only decides where the Redis producer lives, never touches the sink side

- `RedisServiceEvents` (`litellm/redis/transport/events.py`, a frozen dataclass holding the `ServiceLogging`) is the only place REDIS service events for Python traffic are emitted. Its one method, `observe(call_type, call)`, times the call, logs a failure with the caller's service target, emits one success or failure event in the caller's task, and returns the result unchanged. The transport calls it with the method name as `call_type`. Stores and callers emit nothing
- When no `service_callback` is configured, which is the default, `observe` runs the call and logs the failure without walking the stack or building an event
- Rust never calls back into Python per Redis operation. A per-op host call costs a tokio task, a blocking thread, the GIL and an event loop per op, and buffering plus a flusher loses the OTel parent and needs a background task the SDK does not have
- A native store holds the same `RedisServiceEvents` and runs each operation through `observe`, so native traffic reports from Python exactly like transport traffic, with the store operation name as `call_type`
- The breaker for the native client is `RedisConfig::breaker` in Rust. A native store never holds a Python `RedisCircuitBreaker`, one breaker per socket. Breaker transitions are rare and may reach Python through one host call that folds the Rust breaker's state into `redis_breaker_state_counts()`
- Keep metric names and label names as they are. Dashboards alert on `litellm_redis_*`. The one accepted label value change is `call_type`, which becomes the transport method name when a keyspace moves, and `prometheus_services.py` writes it into `function_name`

## Asked to port a store to Rust

Confirm with the user before the first port, since the bridge for it does not exist yet

- The keyspace's module in `litellm-rust/crates/redis-store` implements the same operations as async fns on `crates/redis-client`. Its `AGENTS.md` has the crate rules. Rust callers use the same module, so there is one implementation per operation
- One `#[pyclass]` per store in `crates/python-bridge/src/redis/`, built on one shared native client handle, so Python opens no extra connection per store
- A `Native<Keyspace>Store` in the Python module implements the same `Protocol` over the pyclass, each operation through `RedisServiceEvents.observe`, so callers do not change. The implementation is picked at construction, and the transport-backed one stays for config the native client does not support yet (Sentinel, cloud auth, TLS cert options)
- Results come back as frozen Pydantic models generated from the Rust types, so callers keep matching on the same classes. A decode error maps to `RedisFailure(kind="decode")`
- The codec writes the exact bytes Python writes. `json.dumps` escapes non-ASCII and `serde_json` does not, so a Rust pod could not release a lock a Python pod took
- Each port ships cross-implementation tests against one Redis, Python writes and Rust reads and the reverse, covering keys, values, expiry and atomicity through a rolling deploy

## Tests

- Test a store through its caller against `fake_redis_cache(server)` from `tests/_support/fake_redis.py`, a real `RedisCache` on a `fakeredis.FakeServer`, so namespace, breaker and failure mapping are the production ones. No mocks of our own code
- Cover each result the caller acts on, that the TTL lands (`pttl` on a raw `FakeAsyncRedis`), and an outage (`server.connected = False`)
- When a keyspace moves, seed with the old writer and read with the new reader, and the reverse, so old and new pods interoperate
- Assert returned values and Redis state, never the commands sent
- Rust store tests run against real Redis from `litellm-redis-testing` on both topologies

## Decisions from the first slices

The first transport and store slices settled most of the shape. Keep what is listed under "Keep" as is, and rework anything that still has a "Change" item before extending it

Keep

- `RedisCache.transport()` builds the transport over the cache's own client, namespace and breaker, no second connection, no second breaker
- `RedisFailure` as a value with kinds `transport`, `server`, `decode`, `breaker_open` and a `timeout` flag, and a failed write as an unknown outcome
- Namespace applied once inside the transport, TTL from the caller as a `timedelta` sent in milliseconds, replies as undecoded bytes
- The two steps around every redis-py command, the breaker and the events, one REDIS service event per command, none for `breaker_open`, one WARNING per failure through the timeout throttle with the caller's `with_service_target` in the line
- `call_type` becoming the transport method name (`get`, `set`, `delete`) when a caller moves off `async_*_cache`
- The store shape, a `Protocol` of domain operations plus a frozen dataclass over the transport that owns keys, codec, expiry, Lua and decoding, with policy in the caller
- Byte for byte wire compatibility on a move, including the response polling double `json.dumps`, proven by seeding with the old writer and reading with the old reader
- Tests through the caller on `fake_redis_cache(server)`, asserting values and Redis state, never commands sent
- The Rust store crate shape, async fns on `crates/redis-client` returning domain types, one pyclass per store, a `Native<Keyspace>Store` behind the same `Protocol` picked at construction, frozen Pydantic results, cross-implementation tests on one Redis

Change

- Command dataclasses (`Get`, `Set`, `Delete`) run through an overloaded `execute` with a `match` per backend. With redis-py as the only backend they only add a second place to edit per command. The transport is plain methods

```python
# before
await transport.execute(Set(key, value, ttl=ttl, nx=True))
await transport.execute(Get(key))

# after
await transport.set(key, value, ttl=ttl, nx=True)
await transport.get(key)
```

- A pyo3 `NativeRedisTransport` over the Rust client behind the same Protocol. It mixes the stacks and would have to grow toward redis-py's command set. The Rust path is store level only
- "Native stores skip the Python breaker and events, so both have to come from Rust first". Half right. The breaker comes from Rust, the events do not

```python
@dataclass(frozen=True, slots=True)
class NativeLockStore:
    native: NativeLockStoreHandle   # pyclass on the shared Rust client, breaker inside
    events: RedisServiceEvents      # no Python breaker here, one breaker per socket

    async def acquire(self, key: str, owner: str, *, ttl: timedelta) -> bool | RedisFailure:
        return await self.events.observe("acquire", lambda: _reply(self.native.acquire(key, owner, ttl_ms(ttl))))
```

- A Rust `RedisEvents` implementation that reports every native call to `ServiceLogging` through a host call. Delete the per-call path and keep only the breaker transition
- A transport framed as "the backend swaps to the Rust client without touching callers". The transport is the redis-py seam, nothing else goes behind it
- The events step walking the stack for `caller` and spawning a task per command whether or not a sink is configured. Check `ServiceLogging.has_sinks()` first
- The names `Guard`, `BreakerGuard` and `ReportingGuard`, inherited from the `_redis_circuit_breaker_guard` decorator. Only the breaker guards anything. The breaker step is a function, `run_under_breaker(breaker, name, admit, call)`, which wraps `_run_under_circuit_breaker` and maps its exceptions to `RedisFailure`, and the events step is `RedisServiceEvents.observe`. The transport holds the breaker, the cache's `_admit_async_call` and the events as fields and there is no generic wrapper Protocol between them. Reimplementing the breaker admission in the transport, with `allow()` and a bare `record_failure()`, drops the generation check and the timeout class and is not an option
- Lua through `async_register_script`, a per-loop executor cached in `in_memory_llm_clients_cache`. `transport.eval` takes the script text and does `EVALSHA` with `EVAL` fallback, no registration and no per-loop state
- Lists in replies. `mget` returns a tuple with one slot per key and `eval` returns `int | bytes | None | tuple[...]`, a Lua `false` as `None`

## Scope

- Python moves ship first. Every raw redis-py keyspace outside `litellm/caching/` goes behind a store on the transport before anything under `litellm-rust/` changes for Redis. The first Rust port is the pod lock keyspace, since it is low traffic and its owner-checked release proves both stacks write the same `json.dumps` encoded owner
- A keyspace moves whole in one commit, every operation the caller runs on those keys. Moving the read and leaving the write on `RedisCache` is not a move
- Do only what was asked. An extraction is not a store move, a store move is not a Rust port, and a Rust port of one keyspace does not switch another
- A Rust store that already exists does not by itself justify switching its Python caller. Switching callers is its own change with live verification
