For adding Redis access or moving raw Redis access here, follow [.agents/skills/redis-access/SKILL.md](../../.agents/skills/redis-access/SKILL.md)

- `transport/contract.py` is the Python seam to redis-py. `RedisTransport` is plain methods covering every Redis use in Python, data commands, batches, pub/sub and admin, command shaped and never redis-py shaped. It is not a seam for the Rust client, native stores sit behind a store `Protocol` instead
- `transport/client.py` holds the redis-py client shapes the transport drives, as Protocols with no redis import. `transport/batch.py` is the one pipeline engine, shared by transport batches and `RedisBatch`
- Rules shared with `crates/redis-client`, so Python and Rust pods agree on one Redis
  - The namespace is applied here, once, including Lua `KEYS` and pub/sub channels. A key that already starts with `{namespace}:` is left alone
  - Every write takes its TTL from the caller, `None` means no expiry, there is no default
  - Replies are undecoded bytes, `mget` and `eval` arrays are tuples
  - Failures are values. `transport` covers timeouts and connection loss, where the command may still have applied. `server` means Redis answered with an error and the connection is fine
- `transport/redis_py.py` is the only module here that imports redis-py. `transport/breaker.py` reuses `RedisCache`'s breaker admission and health classification, it never reimplements them
- `transport/events.py` is the only place REDIS service events for transport traffic are emitted. Stores and callers emit nothing and do not log Redis health
- `stores/` holds one module per keyspace, and a keyspace that has a store is only reached through it. A store is a `Protocol` of domain operations plus a frozen dataclass over `RedisTransport`. It owns its keys, value codec, expiry, Lua and decoding, and callers never see a Redis command
- Keys, value bytes and TTL units in Redis are a wire format shared with older pods and Rust. Never change them in place
