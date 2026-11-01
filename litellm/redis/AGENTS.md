For adding Redis access or moving raw Redis access here, follow [.agents/skills/redis-access/SKILL.md](../../.agents/skills/redis-access/SKILL.md)

- `transport/contract.py` is the command-shaped seam. Keep it one command dataclass per Redis command the Python code uses, run through `execute`, never redis-py shaped, so `litellm-rust/crates/redis-client` can sit behind it unchanged
- Rules shared with `crates/redis-client`, so the two implementations agree
  - The namespace is applied here, once. A key that already starts with `{namespace}:` is left alone
  - Every write takes its TTL from the caller, `None` means no expiry, there is no default
  - Replies are undecoded bytes
  - Failures are values. `transport` covers timeouts and connection loss, where the command may still have applied. `server` means Redis answered with an error and the connection is fine
- `transport/redis_py.py` is the only module that touches redis-py's client. The breaker and the exception mapping come from `RedisCache` through the injected guard
- Every backend runs each command through a `Guard` that emits one REDIS service event and logs Redis health failures, so swapping backends never drops metrics. Callers don't log Redis health
- `litellm/caching` imports `litellm/redis`, never the reverse
- `stores/` holds one module per keyspace, and a keyspace that has a store is only reached through it. A store is a `Protocol` of domain operations plus implementations of it. It owns its keys, value codec, expiry, Lua and decoding, does its own IO, and callers never see a Redis command
- The Python implementation runs on any `RedisTransport`. A native implementation wraps the keyspace's `crates/redis-store` module and runs on `crates/redis-client` directly, never through `RedisTransport`
- Keys, value bytes and TTL units in Redis are a wire format shared with older pods and Rust. Never change them in place, see `crates/redis-store/AGENTS.md`
