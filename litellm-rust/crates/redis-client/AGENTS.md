- The one async Redis client for the gateway and the Rust cache crates. `cache-redis` (the response cache behind `python-bridge/src/cache/native`) still opens its own r2d2 pool from a URL, without TLS options, cloud auth, Sentinel or cluster settings, until that keyspace ports onto this client. It owns connecting, timeouts, error classes and the key namespace
- It knows nothing about domains, so key builders, TTL policy, value formats and fallback decisions stay in the crate that owns the state. Every write takes its TTL from the caller, and there is no default
- Async only. Build one `RedisClient` per Redis instance at the process boundary and clone it. Never add a pool, a sync API, `spawn_blocking`, or a client built inside a route
- Every connection sets TCP_NODELAY, like redis-py's `socket.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)`. Set every redis-rs timeout and retry explicitly, citing the redis-py 5.3.1 or `litellm/` source next to each constant. Its defaults (500ms response, 1s connect, 6 reconnect retries) are not the Python proxy's. `Timeouts` takes `socket_timeout` and `socket_connect_timeout`, and the connect timeout falls back to the socket timeout like redis-py. Standalone reconnects on the next command with no retry of the failed one, matching redis-py's `Retry(NoBackoff(), 0)`. Cluster keeps redis-py's 3 `cluster_error_retry_attempts` and waits a flat 250ms between them, which is redis-py's CLUSTERDOWN sleep. redis-py's jittered per-node reconnect backoff has no redis-rs equivalent. The retry set also differs, redis-rs retries LOADING and READONLY internally, while redis-py 5.3.1 raises them. Do not claim exact retry parity
- Resolving config from `config.yaml` / `REDIS_*` env the way `litellm/_redis.py` does belongs to the consumer that reads that config, not here. `RedisConfig` takes already-resolved values

## Errors
- `Transport`, `Server`, `Decode`, each wrapping the redis-rs error as `#[source]`. `ConnectError` is separate because only `connect` returns it
- Classify with `Error::classify(error, deployment)`, never a bare `From<RedisError>`, because redis-py (5.3.1) decides by topology. `Transport` is what Python's `_redis_health_error_types` counts: LOADING everywhere, CLUSTERDOWN and MASTERDOWN only on cluster (only `ClusterParser` maps them), plus NOAUTH, WRONGPASS and the `ERR` messages redis-py maps to `ConnectionError`, matched exactly like `parse_error`. TRYAGAIN is `Server`, since the cluster client already retried it. Each code gets its own rstest case built from a parsed RESP error
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
- Atomic writes require scripting permission (`EVAL`, `EVALSHA`, `SCRIPT LOAD`) and permission for the underlying write, `PTTL`, `PEXPIRE` and `GET` for integer increments. There is no non-atomic fallback for restricted ACLs. Ordinary reads and SET do not require scripting
- Integer increments read their exact decimal result inside the script before expiry, without converting through Lua's double. The public method and pipeline return an exact integer

## Scripts and pipelines
- A `Script` declares every key it touches in KEYS. On cluster an undeclared key works or fails depending on which node owns its slot, so it can pass tests and break in production
- `eval` goes through redis-rs `invoke_async`, which loads the script on NOSCRIPT. Inside a pipeline a script is sent as `EVAL` with its source, so it can never miss the script cache and run out of order with the ops around it
- A pipeline is never atomic. Every op gets its own `Result`, from `.ignore_errors()`, so one rejected op leaves the others alone. A transport failure fails every op of that round trip
- redis-rs rejects an async cluster pipeline whose ops span slots. `Pipeline` groups ops by the slot of their first key and runs the slots concurrently, returning results in input order. Within a slot, runs of idempotent ops (GET, SET, DEL, EXPIRE) share a request and every script op gets its own, sent in order. redis-rs retries a cluster request whole, so a script op batched with others would be applied twice when only a neighbour failed
- Pipeline ops take one key each. A multi-key op would span slots inside one group
- Command builders live in `commands.rs` and are shared by `RedisClient` and `Pipeline`, so a command is written once

## Tests
- Against real Redis from `litellm-redis-testing`, never a mocked wire or asserted command bytes
- Tests that only need the public API live in `tests/<module>.rs`, with client builders in `tests/support`. Inline `#[cfg(test)]` modules are only for private items (`error.rs`, `namespace.rs`)
- Observe connections from the server side (`RedisDeployment::clients`, `cli_on_every_node`), never by reaching into `RedisClient`'s connection
- Every command test runs on each topology through `#[values(Topology::Standalone, Topology::Cluster)]`. A test is single-topology only when the behavior exists on one (a `db` number has no cluster meaning)
- Assert values and Redis state. Outages come from the container (`restart`, `freeze` / `thaw`), not `DEBUG`, which the official image disables
- Use a short client timeout in outage tests so they stay fast. Never sleep to wait for a state; poll with a deadline
