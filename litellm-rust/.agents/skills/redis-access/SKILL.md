---
name: redis-access
description: Add Redis-backed operations, hide existing redis-py access behind typed interfaces, or explicitly port implementations to Rust without expanding migration scope
---

# Redis access

## Start from RedisCache

- Evolve `litellm/caching/redis_cache.py`'s `RedisCache` as the existing caller-facing facade
- It is not yet a fully typed public contract
  - `BaseCache` leaves common operations untyped
  - `_AsyncRedisCommands` is a private typing aid and still exposes redis-py pipelines
  - `init_async_client()`, `init_pubsub_client()`, client attributes, and script callbacks leak implementation details
- Add or strengthen typed operation methods on `RedisCache` before creating another abstraction
- Migrate callers to those methods, then make client access private to the implementation
  - Include sync callers, pub/sub, script users, diagnostics, and shutdown paths
  - Check external compatibility before removing public methods or attributes, deprecating them separately if needed
- Do not require a new protocol or facade for each operation
- Keep domain workflows in their owners, `RedisCache` should not absorb rate-limit or authentication policy

## Adding new Redis-backed access

- Callers depend on typed operations, never redis-py
- Prefer existing `RedisCache` operations, adding the missing typed capability there when appropriate
- Use a separate state interface only when the operation belongs to a domain owner
  - Express the caller's need, such as reserving capacity or draining pending updates
  - Use a narrow Python `Protocol` when an injected dependency needs a contract
  - Do not mirror the whole redis-py API or grow a universal Redis wrapper
- Give callers typed inputs, results, and backend-neutral failures
  - Do not expose clients, pools, pipelines, script handles, Redis reply objects, or redis-py exceptions
  - No `get_client()`, generic `execute_command()`, or callback receiving a raw client
  - A protocol returning a redis-py pipeline still leaks the implementation
- Keep redis-py imports, command execution, reply decoding, and error translation inside the implementation
- Keep key layout, value formats, TTL policy, and fallback decisions with the owning cache or state component
  - The transport applies the namespace once and executes the supplied TTL
- redis-py is fine as the implementation, adding an operation does not require a Rust port

## Asked to move existing access behind an interface

- Find the calls being moved, including raw client attributes, client factories, script registrations, and exception handling
- Identify the complete operation and its atomicity boundary before extracting it
- Move the existing redis-py implementation behind the relevant operation
  - Keep scripts and transactional command sequences together
  - Move script registration and invocation together, not just client construction
- Replace direct client access with the operation and inject the dependency
- Preserve observable behavior with focused regression tests
  - Keys, namespaces, value encoding, TTL units and refresh behavior
  - Ordering, missing values, partial failures, and atomicity
  - Existing fail-open or fail-closed policy and caller-visible errors
- Translate errors at the implementation boundary without hiding an outage as a cache miss
- Keep this an extraction, not a backend migration or unrelated cleanup
- Leave unrelated legacy access unchanged, but add no new raw-client escape hatches

## Example, a read that must distinguish missing from unavailable

- The single-use guard in `litellm/proxy/_experimental/mcp_server/gateway_dcr_flow.py` bypasses the cache getter for this reason
- Before, the caller gets the client, applies the namespace, and interprets Redis failures

```python
async def marker_state(cache: RedisCache, key: str) -> KeyPresence:
    try:
        client = cache.init_async_client()
        value = await client.get(cache.check_and_fix_namespace(key=key))
    except Exception:
        return "unavailable"
    return "missing" if value is None else "present"
```

- After, add a typed `RedisCache.async_key_presence(key: str) -> KeyPresence` operation
  - `KeyPresence = Literal["present", "missing", "unavailable"]`
  - These are proposed additions, not APIs implemented by this documentation PR

```python
async def marker_state(cache: RedisCache, key: str) -> KeyPresence:
    return await cache.async_key_presence(key)
```

- The implementation owns the namespaced GET, failure translation, and diagnostics
  - Preserve raw GET semantics, including wrong-type failures and stored JSON `null` counting as present
  - Preserve the caller's existing fail-closed decision on `"unavailable"`
- Do not substitute `async_get_cache()` unchanged, its default hides failures as misses and decodes values
  - `raise_on_error=True` alone does not give a typed result or hide redis-py exceptions
- Test a missing key, a stored value, stored JSON `null`, a wrong-type key, and an unavailable Redis
- Apply the same pattern to scripts and transactions, expose a typed operation rather than returning an executor or pipeline

## An equivalent Rust operation already exists

- Leave it and the current backend selection unchanged unless integration is explicitly requested
- Its existence alone does not justify switching a Python caller to Rust
- Interface extraction can keep using redis-py even when the Rust operation is available

## Asked to port an implementation to Rust

- Reuse `crates/redis-client` for transport and follow its `AGENTS.md`
  - Keep domain operations, keys, value formats, TTL policy, and fallback outside the transport crate
  - Keep redis-rs types and raw replies out of domain-facing and Python-facing contracts
- Resolve configuration in `crates/config` and compose clients at the gateway boundary
  - Preserve coordination, transaction-buffer, and response-cache role selection
  - Gateway client construction alone does not wire request consumers
- Keep Python conversion and native-handle construction at the Python bridge boundary
- Switch callers through the same operation interface only when integration is requested
- Verify equivalent behavior against real Redis before switching callers
  - Check supported standalone and cluster topologies, slot placement, and namespace compatibility
  - Check TTLs, encoding, missing values, concurrency, errors, and outage behavior
  - Rust pipelines are non-atomic, never substitute one for a transaction or atomic script
- Never retry an ambiguous write through another backend, a timeout may follow a committed increment, pop, or lock
- Remove the old implementation only when no callers need it

## Migration direction

- Centralize access behind typed operations first, then replace implementations incrementally
- Callers should not change when redis-py is replaced by Rust
- Inject interface fakes for caller tests and test implementations against real Redis
- Keep client lifecycle, connection tuning, and topology handling inside infrastructure implementations
- Mixed Python and Rust deployments must agree on shared keys, namespaces, value formats, and atomic operations
- Redis client configuration, caller migration, and redis-py removal are separate steps
