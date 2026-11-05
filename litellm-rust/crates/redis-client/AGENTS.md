- The one async Redis client for the gateway and the Rust cache crates. It owns connecting, timeouts, error classes and the key namespace
- It knows nothing about domains, so key builders, TTL policy, value formats and fallback decisions stay in the crate that owns the state. Every write takes its TTL from the caller, and there is no default
- Async only. Build one `RedisClient` per Redis instance at the process boundary and clone it. Never add a pool, a sync API, `spawn_blocking`, or a client built inside a route
- Set every redis-rs timeout and retry explicitly. Its defaults (500ms response, 1s connect, 6 reconnect retries) are not the Python proxy's (5s socket timeout from `REDIS_SOCKET_TIMEOUT`)
- Resolving config from `config.yaml` / `REDIS_*` env the way `litellm/_redis.py` does belongs to the consumer that reads that config, not here. `RedisConfig` takes already-resolved values

## Errors
- `Transport`, `Server`, `Decode`, each wrapping the redis-rs error as `#[source]`. `ConnectError` is separate because only `connect` returns it
- Classify in `From<RedisError> for Error` by `ErrorKind`. LOADING, CLUSTERDOWN, MASTERDOWN, TRYAGAIN and READONLY are typed server replies, but they mean the node can't serve right now, so they are `Transport`. Codes redis-rs doesn't type (WRONGTYPE) come back as `ErrorKind::Extension` and are `Server`. Each code gets its own rstest case built from a parsed RESP error
- A `Server` error never drops the connection
- `Transport` includes timeouts, where the command may still have applied. Callers of counters and locks treat it as an unknown outcome
- Never swallow an error or turn it into a miss. The caller picks the fallback
- Reads return `Option<T>`: a missing key is `None`, and a value that doesn't decode to `T` (`"15.0"` as `i64`) is `Decode`

## Circuit breaker
- The `litellm-circuit-breaker` `Breaker` is injected through `RedisConfig::breaker` and shared by every clone. `None` turns it off
- Every public call (each command, `eval`, one `run` of a pipeline) takes one admission through `guarded` / `admit` in `client.rs`. Never call the connection around it
- `Transport` counts as a failure (`Timeout` when `is_timeout()`, else `Hard`). `Server` and `Decode` count as nothing, since Redis answered and the request was bad. A probe that gets one still reopens, through the dropped-probe rule in `circuit-breaker`. A pipeline counts one failure if any op came back `Transport`, else one success
- An open breaker returns `Error::BreakerOpen` without touching the socket
- Python doesn't count MASTERDOWN, TRYAGAIN or READONLY as health failures. Here they are `Transport` and do count

## Namespace
- Applied once, here, to every key. A key that already starts with `{namespace}:` is left alone, matching `RedisCache.check_and_fix_namespace`
- The rule never changes, because Python and Rust pods share keys in the same Redis

## Commands
- One typed method per command, built with `redis::cmd`. Empty batches (MGET, DEL) return without a round trip, since Redis rejects an empty MGET
- On cluster, redis-rs splits MGET and DEL by slot and returns MGET values in input order. Other multi-key commands are not split, so a new one needs a cross-slot test
- TTLs are `Duration` and are sent in milliseconds (`PX`, `PEXPIRE`, `PTTL`)

## Scripts and pipelines
- A `Script` declares every key it touches in KEYS. On cluster an undeclared key works or fails depending on which node owns its slot, so it can pass tests and break in production
- `eval` goes through redis-rs `invoke_async`, which loads the script on NOSCRIPT. Inside a pipeline nothing reloads, so a NOSCRIPT op reruns alone with `invoke_async`
- A pipeline is never atomic. Every op gets its own `Result`, from `.ignore_errors()`, so one rejected op leaves the others alone. A transport failure fails every op of that round trip
- redis-rs rejects an async cluster pipeline whose ops span slots. `Pipeline` groups ops by the slot of their first key, runs the groups concurrently and returns results in input order
- Pipeline ops take one key each. A multi-key op would span slots inside one group
- Command builders live in `commands.rs` and are shared by `RedisClient` and `Pipeline`, so a command is written once

## Tests
- Against real Redis from `litellm-redis-testing`, never a mocked wire or asserted command bytes
- Every command test runs on each topology through `#[values(Topology::Standalone, Topology::Cluster)]`. A test is single-topology only when the behavior exists on one (a `db` number has no cluster meaning)
- Assert values and Redis state. Outages come from the container (`restart`, `freeze` / `thaw`), not `DEBUG`, which the official image disables
- Use a short client timeout in outage tests so they stay fast. Never sleep to wait for a state; poll with a deadline
