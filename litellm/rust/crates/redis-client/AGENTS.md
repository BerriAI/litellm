- The one async Redis client for the gateway and the Rust cache crates. It owns connecting, timeouts, error classes and the key namespace
- It knows nothing about domains, so key builders, TTL policy, value formats and fallback decisions stay in the crate that owns the state. Every write takes its TTL from the caller, and there is no default
- Async only. Build one `RedisClient` per Redis instance at the process boundary and clone it. Never add a pool, a sync API, `spawn_blocking`, or a client built inside a route
- Every connection sets TCP_NODELAY, like redis-py's `socket.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)`. Set every redis-rs timeout and retry explicitly, citing the redis-py 5.3.1 or `litellm/` source next to each constant. Its defaults (500ms response, 1s connect, 6 reconnect retries) are not the Python proxy's. `Timeouts` takes `socket_timeout` and `socket_connect_timeout`, and the connect timeout falls back to the socket timeout like redis-py. Standalone reconnects on the next command with no retry of the failed one, matching redis-py's `Retry(NoBackoff(), 0)`. redis-rs keeps a socket whose command timed out, so `Renewing` in `connection.rs` swaps in a fresh `ConnectionManager`, or a fresh cluster connection like redis-py's cluster `aclose`, after a timeout. Without it a socket that went silent times out every command until TCP gives up on it. Cluster keeps redis-py's 3 `cluster_error_retry_attempts` and waits a flat 250ms between them, which is redis-py's CLUSTERDOWN sleep. redis-py's jittered per-node reconnect backoff has no redis-rs equivalent. The retry set also differs, redis-rs retries LOADING and READONLY internally, while redis-py 5.3.1 raises them. Do not claim exact retry parity
- Resolving config from `config.yaml` / `REDIS_*` env the way `litellm/_redis.py` does belongs to the consumer that reads that config, not here. `RedisConfig` takes already-resolved values

## Errors
- `Transport`, `Server`, `Decode`, each wrapping the redis-rs error as `#[source]`. `ConnectError` is separate because only `connect` returns it
- Classify with `Error::classify(error, deployment)`, never a bare `From<RedisError>`, because redis-py (5.3.1) decides by topology. `Transport` is what Python's `_redis_health_error_types` counts: LOADING everywhere, CLUSTERDOWN and MASTERDOWN only on cluster (only `ClusterParser` maps them), READONLY only under Sentinel (where `SentinelManagedConnection` turns it into `ConnectionError` and rediscovers), plus NOAUTH, WRONGPASS and the `ERR` messages redis-py maps to `ConnectionError`, matched exactly like `parse_error`. TRYAGAIN is `Server`, since the cluster client already retried it. Each code gets its own rstest case built from a parsed RESP error
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
- A command is a value, one constructor in `commands.rs` returning `Command<T>`: its alone and pipelined forms and its reply decode. `RedisClient::send`, `Pipeline::with` and `RedisBatch::add` are the only entry points, and each applies its namespace to the keys once. A new command is one constructor, never a method on an executor. `ping` and `subscribe` stay methods. Empty batches (MGET, DEL) return without a round trip, since Redis rejects an empty MGET
- A store attaches its value codec with `Command::map`, never a keyspace codec inside this crate
- On cluster, redis-rs splits MGET and DEL by slot and returns MGET values in input order. Other multi-key commands are not split, so a new one needs a cross-slot test
- Every command is the one `RedisCache` sends, argument for argument, so `MONITOR` on a mixed fleet shows one shape. A difference from main's Python is a defect even when the result is the same
- TTLs are `Duration` and are sent in whole seconds (`SET ... EX`, `EXPIRE`, `TTL`), like redis-py's `int(ttl)`. `set_nx` sends `EX` before `NX`, the order redis-py builds
- Numbers go out as `Amount`, written the way redis-py encodes them: an `int` with `str()` and a `float` with `repr()` from `python-compat`
- Increments are plain commands, never a script. `incr_by` is `increment_cache`: `INCRBY`, `TTL`, then `EXPIRE` only on a key with no TTL, each its own round trip. `incr_by_float` alone is `async_increment`: `INCRBYFLOAT` with `EXPIRE` (`Expiry::Refresh`) or `TTL` (`Expiry::KeepExisting`) in one round trip, and for `KeepExisting` a third, `EXPIRE`, only on a key with no TTL. Pipelined it is `RedisBatch`'s `_Increment`, `INCRBYFLOAT` then `EXPIRE` when an expiry is given, `KeepExisting` lowered to `Refresh`. `Plan` keeps only the cases whose alone and pipelined forms differ

## Scripts and pipelines
- A `Script` declares every key it touches in KEYS. On cluster an undeclared key works or fails depending on which node owns its slot, so it can pass tests and break in production
- `eval` goes through redis-rs `invoke_async`, which loads the script on NOSCRIPT. Inside a pipeline a script is sent as `EVALSHA`, like `RedisBatch`. A `NOSCRIPT` reply means that op ran nothing, so it alone runs again after the pipeline through `invoke_async` (`EVALSHA`, `SCRIPT LOAD`, `EVALSHA`), like `RedisBatch`'s `_settle_alone`. It then lands after the ops around it, as it does in Python
- `Plan::Script` keeps a script call apart from plain commands, and the one resend takes only a `ScriptCall` whose reply was `NOSCRIPT`. Nothing can resend a pipeline or a command that already ran, so a retry can never apply an increment twice
- `RedisBatch::pipelined` is one `Pipeline` whose ops answer as typed `Declared<T>`s. A store that sends its own pipeline uses it rather than decoding `run`'s raw `Value`s
- A pipeline is never atomic. Every op gets its own `Result`, from `.ignore_errors()`, so one rejected op leaves the others alone. A transport failure fails every op of that round trip
- redis-rs rejects an async cluster pipeline whose ops span slots. `Pipeline` groups ops by the slot of their first key and runs the slots concurrently, returning results in input order. Within a slot, runs of idempotent ops (GET, SET, DEL, EXPIRE) share a request and every script op gets its own, sent in order. redis-rs retries a cluster request whole, so a script op batched with others would be applied twice when only a neighbour failed
- Pipeline ops take one key each. A multi-key op would span slots inside one group. An op can send more than one command for that key (`INCRBYFLOAT` then `EXPIRE`), and settles from the first reply, failing if any of its replies failed
- Command constructors live in `commands.rs` and every executor sends them, so a command is written once

## Pub/sub
- Classic `PUBLISH` / `SUBSCRIBE`, never sharded, since redis-py publishes and subscribes through one plain client and cluster broadcasts classic messages to every node. Payloads are raw bytes, and the codec belongs to the store that owns the channel
- Channel names never get the namespace. Python builds `{namespace}:{channel}` itself, so the store does too
- `publish` is a command like any other, guarded by the breaker and reported to events
- `subscribe` returns a `Stream` that opens its own RESP2 `PubSub` connection when first polled, like redis-py's `PubSub`, to the endpoint, the first reachable cluster node, or the master Sentinel names at that moment. It never touches the breaker or events, like Python's subscriber, which reads off its own connection
- After a lost connection or a failed subscribe it waits and resubscribes, 5s doubling to 60s and reset by a successful subscribe, like `AuthCacheInvalidationSubscriber._run`. Messages published while it is down are lost. The connection has no idle ping, so a socket that dies silently is only noticed by TCP
- redis-rs's `PubSub` takes no credentials provider, so a provider is asked for its current credentials once when the subscription connects, which is when redis-py's pubsub connection asks it. The connect is bounded by the connect timeout

## Batches
- `RedisBatch` is litellm's `RedisBatch` (`litellm/caching/redis_batch.py`). It keeps each declared op as data and, on a standalone or Sentinel Redis, sends them later as one `Pipeline`, so it inherits every pipeline rule above. On a cluster it never pipelines and runs each op on its own, concurrently, the way `_settle_alone` calls the `RedisCache` method behind it. An increment there keeps a TTL the key already has, as `async_increment` does, where the pipelined one refreshes it like `_Increment`
- `mget` is one `MGET` of the keys in order with repeats read once, like `_MGet`, and answers one value per key it was given, in that order. Use it instead of one `get` per key whenever Python reads several keys in one call
- It knows nothing about requests. The host that owns a request (Axum handler, pyo3 adapter) builds one and passes it down as a value, never through a task-local
- `RedisBatch::alone` runs every op on its own like the cluster path, so a fresh alone batch is "send it now" and `Declared<T>` is the one reply handle on every route
- Each op returns a `Declared<T>` backed by a oneshot. `flush` sends everything declared so far, one flush at a time, and an empty flush makes no round trip. `resolve` flushes only when its op has not gone out yet, so an op declared after another caller resolved costs a second round trip. Stages that know their reads up front declare all of them and call `flush` once
- An op whose flush was cancelled resolves as `Transport`, because it may already have reached Redis
- `settled` waits for whichever flush carries the op and never sends one itself, for follow-ups that must not force a flush. `RedisClient::same_client` tells clones of one client apart from a second connection, so a host can keep one batch per client
- Against real Redis from `litellm-redis-testing`, never a mocked wire or asserted command bytes
- Tests that only need the public API live in `tests/<module>.rs`, with client builders in `tests/support`. Inline `#[cfg(test)]` modules are only for private items (`error.rs`, `namespace.rs`)
- Observe connections from the server side (`RedisDeployment::clients`, `cli_on_every_node`), never by reaching into `RedisClient`'s connection
- Every command test runs on each topology through `#[values(Topology::Standalone, Topology::Cluster)]`. A test is single-topology only when the behavior exists on one (a `db` number has no cluster meaning)
- Assert values and Redis state. Outages come from the container (`restart`, `freeze` / `thaw`), not `DEBUG`, which the official image disables. A socket that never answers again comes from `TcpProxy::blackhole_open_connections`, since `thaw` revives the same socket
- Use a short client timeout in outage tests so they stay fast. Never sleep to wait for a state; poll with a deadline

## TLS, credentials and Sentinel
- `TlsOptions` follows redis-py 5.3.1. It verifies the certificate by default, skips the hostname check by default (`ssl_check_hostname=False`), and trusts the system roots plus `ca_certs`
- rustls only reads unencrypted keys, so a PKCS#8 `ENCRYPTED PRIVATE KEY` is decrypted with `key_password` before it reaches redis-rs. A legacy `Proc-Type: 4,ENCRYPTED` PEM key is refused with `KeyError::LegacyEncryption` rather than decrypted by hand-rolled OpenSSL key derivation
- redis-rs loads the system trust store for every TLS connection and fails if that fails, even with `ca_certs` set. Where the macOS keychain is blocked (agent sandboxes), run the TLS tests with `SSL_CERT_FILE=/etc/ssl/cert.pem`
- `Credentials::Static` are sent at connect, and URL userinfo wins over them, like redis-py's `from_url`
- A url's `?protocol=` is ignored and every connection speaks RESP2, which every reply decode assumes
- `Credentials::Provider` wraps a redis-rs `StreamingCredentialsProvider`. redis-rs takes the first item at connect and re-`AUTH`s live connections on every later item, on each cluster node. Providers push, redis-rs never polls. With a provider, redis-rs overwrites URL userinfo at connect, which matches Python stripping `user:pass@` from the URL
- Sentinel discovers the master on each new connection and drops it on a transport failure or a READONLY reply, to one command or inside a pipeline, like redis-py's `SentinelConnectionPool`
- redis-rs sends `ROLE` to the master a Sentinel names before using it, which redis-py never does. That probe gets the data credentials (a provider's current pair) and TLS, so the Sentinel client is built on each discovery. redis-rs gives the probe no way to skip the hostname check, and drops the client certificate when the certificate is not verified, so a TLS master whose certificate does not name the announced address, or that requires a client certificate while verification is off, fails discovery where Python connects
- Sentinel monitors get `sentinel_password`, the data username when that password is set, and the TLS mode and certificates. Python's `_sentinel_auth_kwargs` forwards every other connection kwarg too, and it sends the username even without a sentinel password
- A `Debug` impl never shows a secret. Follow `.agents/skills/rust-redaction`
