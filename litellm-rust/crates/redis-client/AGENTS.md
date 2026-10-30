- The one async Redis client for the gateway and the Rust cache crates. `cache-redis` (the response cache behind `python-bridge/src/cache/native`) still opens its own r2d2 pool from a URL, without TLS options, cloud auth, Sentinel or cluster settings, until that keyspace ports onto this client. It owns connecting, timeouts, error classes and the key namespace
- It knows nothing about domains, so key builders, TTL policy, value formats and fallback decisions stay in the crate that owns the state. Every write takes its TTL from the caller, and there is no default
- Async only. Build one `RedisClient` per Redis instance at the process boundary and clone it. Never add a pool, a sync API, `spawn_blocking`, or a client built inside a route
- Set every redis-rs timeout and retry explicitly. Its defaults (500ms response, 1s connect, 6 reconnect retries) are not the Python proxy's. `Timeouts` takes `socket_timeout` and `socket_connect_timeout`, and the connect timeout falls back to the socket timeout like redis-py. Standalone reconnects once per failed command with no backoff retries, matching redis-py 5.3.1's `Retry(NoBackoff(), 0)`. Cluster keeps redis-py's 3 `cluster_error_retry_attempts`
- Resolving config from `config.yaml` / `REDIS_*` env the way `litellm/_redis.py` does belongs to the consumer that reads that config, not here. `RedisConfig` takes already-resolved values

## Errors
- `Transport`, `Server`, `Decode`, each wrapping the redis-rs error as `#[source]`. `ConnectError` is separate because only `connect` returns it
- Classify with `Error::classify(error, deployment)`, never a bare `From<RedisError>`, because redis-py (5.3.1) decides by topology. `Transport` is what Python's `_redis_health_error_types` counts: LOADING and CLUSTERDOWN everywhere, MASTERDOWN only on cluster, READONLY only under Sentinel (where it also triggers rediscovery), plus NOAUTH, WRONGPASS and the `ERR` messages redis-py maps to `ConnectionError`. TRYAGAIN is `Server`, since the cluster client already retried it. Each code gets its own rstest case built from a parsed RESP error
- A `Server` error never drops the connection
- `Transport` includes timeouts, where the command may still have applied. Callers of counters and locks treat it as an unknown outcome
- Never swallow an error or turn it into a miss. The caller picks the fallback
- Reads return `Option<T>`: a missing key is `None`, and a value that doesn't decode to `T` (`"15.0"` as `i64`) is `Decode`

## Circuit breaker
- The `litellm-circuit-breaker` `Breaker` is injected through `RedisConfig::breaker` and shared by every clone. `None` turns it off
- Every public call (each command, `eval`, one `run` of a pipeline) takes one admission through `guarded` / `admit` in `client.rs`. Never call the connection around it
- `Transport` counts as a failure (`Timeout` when `is_timeout()`, else `Hard`). `Server` and `Decode` count as nothing, since Redis answered and the request was bad. A probe that gets one still reopens, through the dropped-probe rule in `circuit-breaker`. A pipeline counts one failure if any op came back `Transport`, else one success
- An open breaker returns `Error::BreakerOpen` without touching the socket

## Namespace
- Applied once, here, to every key. A key that already starts with `{namespace}:` is left alone, matching `RedisCache.check_and_fix_namespace`
- The rule never changes, because Python and Rust pods share keys in the same Redis

## Commands
- One typed method per command, built with `redis::cmd`. Empty batches (MGET, DEL) return without a round trip, since Redis rejects an empty MGET
- On cluster, redis-rs splits MGET and DEL by slot and returns MGET values in input order. Other multi-key commands are not split, so a new one needs a cross-slot test
- TTLs are `Duration` and are sent in milliseconds (`PX`, `PEXPIRE`, `PTTL`)
- `incr_by`, `incr_by_float` and `rpush` take an `Expiry` and apply it in the same script as the write, so a key never outlives a lost follow-up `EXPIRE`. `KeepExisting` sets the TTL only on a key without one, like Python's `_incrbyfloat_with_ttl`, and `Refresh` always resets it

## Scripts and pipelines
- A `Script` declares every key it touches in KEYS. On cluster an undeclared key works or fails depending on which node owns its slot, so it can pass tests and break in production
- `eval` goes through redis-rs `invoke_async`, which loads the script on NOSCRIPT. Inside a pipeline a script is sent as `EVAL` with its source, so it can never miss the script cache and run out of order with the ops around it
- A pipeline is never atomic. Every op gets its own `Result`, from `.ignore_errors()`, so one rejected op leaves the others alone. A transport failure fails every op of that round trip
- redis-rs rejects an async cluster pipeline whose ops span slots. `Pipeline` groups ops by the slot of their first key, runs the groups concurrently and returns results in input order
- Pipeline ops take one key each. A multi-key op would span slots inside one group
- Command builders live in `commands.rs` and are shared by `RedisClient` and `Pipeline`, so a command is written once

## Tests
- Against real Redis from `litellm-redis-testing`, never a mocked wire or asserted command bytes
- Tests that only need the public API live in `tests/<module>.rs`, with client builders in `tests/support`. Inline `#[cfg(test)]` modules are only for private items (`error.rs`, `namespace.rs`)
- Observe connections from the server side (`RedisDeployment::clients`, `cli_on_every_node`), never by reaching into `RedisClient`'s connection
- Every command test runs on each topology through `#[values(Topology::Standalone, Topology::Cluster)]`. A test is single-topology only when the behavior exists on one (a `db` number has no cluster meaning)
- Assert values and Redis state. Outages come from the container (`restart`, `freeze` / `thaw`), not `DEBUG`, which the official image disables
- Use a short client timeout in outage tests so they stay fast. Never sleep to wait for a state; poll with a deadline

## Credentials, TLS and Sentinel
- `Credentials::Provider` wraps a redis-rs `StreamingCredentialsProvider`. redis-rs takes the first item at connect and re-`AUTH`s live connections on every later item, on each cluster node. Providers push, redis-rs never polls
- `TlsOptions` follows redis-py 5.3.1. It verifies the certificate by default, skips the hostname check by default (`ssl_check_hostname=False`), and trusts the system roots plus `ca_certs`
- redis-rs loads the system trust store for every TLS connection and fails if that fails, even with `ca_certs` set. Where the macOS keychain is blocked (agent sandboxes), run the TLS tests with `SSL_CERT_FILE=/etc/ssl/cert.pem`
- Sentinel discovers the master on each new connection and drops it on a transport failure or READONLY, like redis-py's `SentinelConnectionPool`
- Sentinel monitors get `sentinel_password`, the data username when that password is set, and the TLS mode and certificates. Python's `_sentinel_auth_kwargs` forwards every other connection kwarg too, and it sends the username even without a sentinel password
- With `Credentials::Provider`, redis-rs overwrites URL userinfo with the provider's credentials at connect, which matches Python stripping `user:pass@` from the URL
- A token-only credential is sent as `AUTH default <token>`, because redis-rs always sends a username with a provider. On Redis 6+ that equals Python's `AUTH <token>`
- A `Debug` impl never shows a secret. Follow `.agents/skills/rust-redaction`
