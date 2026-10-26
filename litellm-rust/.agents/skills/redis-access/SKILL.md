---
name: redis-access
description: Use and extend RedisCache's provisional typed operations while keeping Redis transport details private and preserving existing behavior
---

# Redis access

## Use the existing interface

`litellm/caching/redis_cache.py` owns the caller-facing `RedisCache` interface. Internal workflows use its typed operations. Their value types live in `litellm/types/caching.py`, transport and reply adapters in `litellm/caching/_redis_capabilities.py`, and atomic scripts in `litellm/caching/_redis_scripts.py`

Choose the existing operation that represents the caller's need. Do not add another Redis facade, a protocol for every operation, or a generic command wrapper

| Need | Existing operations |
| --- | --- |
| Distinguish missing from unavailable | `async_key_presence` |
| Read ordered counters or cached values | `async_read_atomic_counts`, `async_read_values` |
| Maintain request and token windows | `async_increment_window_counters`, `async_reserve_window_capacity`, `async_increment_guarded_counters` |
| Own concurrency slots | `async_acquire_concurrency`, `async_release_concurrency`, `async_count_concurrency` |
| Increment, reserve or refund counters | `async_increment_counters`, `async_reserve_counter`, `async_refund_counter` |
| Store and consume reservation records | `async_save_record`, `async_pop_record`, `async_delete_record` |
| Enqueue serialized updates | `async_enqueue_updates` with `CacheQueueUpdate` |
| Count session iterations or spend | `async_increment_iterations`, `async_increment_spend` |
| Claim affinity | `async_claim_affinity` |
| Read and record login throttling | `async_read_throttle_ttls`, `async_record_throttle_failure` |
| Own a lease | `async_acquire_lease`, `async_extend_lease`, `async_release_lease`, `async_lease_held` |
| Publish or subscribe | `async_publish`, `async_publish_many`, `async_subscribe` |
| Inspect connection capacity | `get_connection_pool_info` |

Most atomic operations return an awaitable `CacheOperation[T]`. Await it to obtain `T | CacheFailure`. Pub/sub and value-read methods are coroutines with explicit result unions. `async_key_presence` returns `KeyPresence`, the literal union `"present" | "missing" | "unavailable"`. Check the method signature rather than assuming that every operation has the same return shape

Legacy cache read/write methods remain available. They are not interchangeable with the new operations when their error handling, encoding or missing-value semantics differ

## Keep policy in the owner

The owner supplies domain keys, values, thresholds, tokens and TTL policy. It decides whether an unavailable cache permits a request, rejects it, or triggers a local fallback. `RedisCache` applies its namespace once, executes the supplied TTL, owns atomic command sequences, validates replies and translates backend errors

For example, the gateway's single-use marker check needs the raw presence distinction, including a stored JSON `null` counting as present and a wrong-type key counting as unavailable

```python
from litellm.caching.redis_cache import RedisCache
from litellm.types.caching import KeyPresence

async def marker_state(cache: RedisCache, key: str) -> KeyPresence:
    return await cache.async_key_presence(key)
```

Do not replace this operation with `async_get_cache`. Its default error handling can hide an outage as a miss, and it decodes stored values. `raise_on_error=True` does not supply a backend-neutral result contract

`CacheFailure.kind` distinguishes unavailable transport from malformed replies. `reason` distinguishes an open circuit, a timeout and other failures. Do not parse error messages or import Redis exceptions in the owner. Preserve the owner's existing malformed-response behavior separately from outage fallback. The optional validation error exists for legacy validation contracts, not as a transport exception

Use `log_redis_failure` when preserving existing Redis failure diagnostics. An open circuit stays at debug level and timeouts retain throttling. Do not turn every failed operation into a warning or error storm

## Batching and subscriptions

Atomic methods that accept `batch=` can share a `RedisBatch`. The cache declares the command internally, and the caller receives the same typed operation result. Declare independent operations before awaiting them to retain batching. The non-transactional batch is not an atomicity boundary. Each Lua operation retains its own atomic boundary, ordered replies and failure isolation

`CacheOperation.done` and `on_settled` support the existing cancellation and settlement workflows. Settlement callbacks receive typed results, never clients, pipelines or script handles. Preserve these hooks when a request can be cancelled after a write has reached Redis

`async_subscribe` returns a `CacheSubscription`. Call `receive(timeout_seconds=...)` for a payload, idle `None`, or `CacheFailure`, and call `close` during shutdown and reconnect cleanup. Redis subscription envelopes, pooling, routing and reconnection transport remain private. The owner retains its reconnect timing, event validation and message application policy

## Provisional does not mean deprecated

The new typed API is provisional. Public operations emit `FutureWarning` with the message `RedisCache.<operation> is experimental and may change in a future release`. Operation lifecycle methods and subscription methods follow the same policy. Keep the warning at the user's call site with the correct stack level and ordinary warning filtering. Do not use a deprecation decorator for these operations and do not globally silence warnings

Legacy raw accessors remain callable for compatibility. `init_async_client`, `init_pubsub_client`, and `async_register_script` retain PEP 702 metadata through `typing_extensions.deprecated(..., category=None)`. Existing raw attributes and compatibility exports remain available to external consumers. `RedisBatch.script` is also a deprecated compatibility surface, not an internal extension API

Only the cache implementation may use the private transport bridge. Its deliberate dispatch through legacy factories preserves subclass overrides and loop-local client reuse. Exact static suppressions there and on compatibility-only re-exports document that narrow exception. They are not permission for another internal caller to use a raw accessor

## Add a missing capability

Start with the complete operation and its observable contract. Add a typed method to `RedisCache` and a narrow immutable result type only if the existing operations cannot express the requirement. A small domain protocol is appropriate for injection when it belongs to an existing owner, but do not mirror redis-py

Keep redis-py imports, raw attributes, connection construction, script registration, Lua execution, pipelines, reply decoding and Redis exception translation inside the cache implementation. Do not expose `get_client`, generic `execute_command`, raw-client callbacks, raw Redis replies or Redis exceptions. A typed protocol that returns a pipeline still leaks the transport

Preserve existing Lua bodies verbatim when moving them. Do not shorten or rewrite scripts during interface extraction. Preserve command order, arguments, namespace and cluster-slot placement, value formats, TTL units and refresh rules, batching, retry behavior, telemetry and cancellation. In particular, window timestamps used by guarded updates must retain their textual identity, including fractional values

Return a typed failure when transport or reply validation fails. Keep missing data distinct from an outage where the owner needs that distinction. Do not retry an ambiguous write through a second backend, because a timeout can follow a committed increment, pop or lock acquisition

Add the provisional warning at the public entry point. Keep implementation helpers private, and never route an internal caller through a compatibility-only raw accessor or alias

## Verify behavior

Extend the existing mapped unit tests. Use in-process transports or injected domain dependencies and assert observable values, namespaces, TTLs, ordering, concurrency ownership, malformed replies, missing keys and outage decisions. Do not mock the new method merely to echo the result being asserted

Test warning category, experimental wording and caller location. Test legacy access under warnings-as-errors and retain subclass dispatch, client reuse and compatibility exports. Audit direct calls, aliased imports, raw attributes, `getattr` and indirect helper paths when checking the boundary. Cache implementation and compatibility-only exports are explicit exemptions. Independent semantic and vector-store Redis integrations are separate from RedisCache

For command-level verification, use the existing Redis integration grouping with isolated keys and cleanup. Exercise atomicity, contention, expiry, pub/sub lifecycle, cluster behavior where supported, and batch partial failures and NoScript retry. Report untested topologies rather than treating standalone Redis as cluster coverage

Run scoped lint and typing comparison gates without changing budgets or ceilings. Avoid broad auto-fixes in legacy caller or test files during extraction

## Backend changes require a separate request

redis-py remains a valid private implementation. An equivalent Rust operation does not authorize switching callers or changing backend selection. Interface extraction, configuration wiring and backend migration are separate work

If explicitly asked to port an implementation, reuse `crates/redis-client` and its `AGENTS.md`, resolve configuration in `crates/config`, and keep native conversion at the Python bridge. Preserve the same typed operations and domain policies. Verify standalone and supported cluster behavior against real Redis before switching. Rust pipelines are not substitutes for atomic scripts or transactions
