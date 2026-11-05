- Redis-backed domain operations shared by the Rust gateway and Python, like a `crates/db` repository. Callers ask for domain operations (`acquire`, `increment`, `get`) and never see a Redis command
- One module per keyspace (rate limit windows, session caps, enqueued tokens, cooldowns, affinity pins, locks, login throttle, spend counters, update buffers, response polling). The module owns its key layout, value codec, expiry, Lua and reply decoding, and nothing outside it builds those keys
- Operations are async fns that take a `&litellm_redis_client::RedisClient` and run their commands on it. The client applies the namespace and the breaker, so a store never does either and never goes back through Python's `RedisTransport`
- Results are domain types (`Acquire::Taken`, `Option<f64>`), never a raw `redis::Value`. A reply that doesn't decode is an error, never a miss or a default count
- Callers keep policy, like which limit applies, fail open or closed, retries and logging

## Wire contract
- Keys, value bytes, TTL units and Lua behavior are a wire format shared with Python pods and older releases on the same Redis. A difference doesn't error, it silently splits counters, budgets or locks between pods during a rolling deploy or rollback
- A keyspace ported from Python keeps the bytes Python writes exactly, including hash tags, `json.dumps` spacing and ASCII escaping, float formatting and seconds vs milliseconds
- Never change a key, codec or TTL unit in place. Write under a new versioned key, read both for one release, then drop the old one, as its own change

## Shape
- Logical keys only, the client adds the namespace. Lua that touches more than one key keeps them in one slot with a `{hash tag}`, and declares every key in KEYS
- Lua is a `&'static str` next to the operation that runs it
- Response-cache identity (normalization, hashing into a logical key) is not store logic. This crate only maps that logical key to storage. Reuse or move `cache-response`'s key builder, never write a third
- Pub/sub, admin commands and vector search are not keyspaces and don't belong here
- Python reaches a store through one `#[pyclass]` per module in `crates/python-bridge/src/redis/`, built with the shared `NativeRedisClient`, behind the keyspace's Python store Protocol in `litellm/redis/stores/`

## Tests
- Run operations against real Redis from `litellm-redis-testing`, on each topology through `#[values(Topology::Standalone, Topology::Cluster)]`. Never assert command bytes or a mocked wire
- Assert decoded results, TTLs and atomicity through the store's own operations and Redis state
- A keyspace ported from Python ships cross-implementation tests against one Redis, where the Python store writes and this crate reads and the reverse
