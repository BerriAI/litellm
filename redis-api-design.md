# Native Redis API design

This covers the layering for the Rust Redis stack behind the Python proxy, from how a Redis command is written down in `redis-client` up to how Python store wrappers pick a request phase. It came out of reviewing the PyO3 store bindings in the `litellm_router_stores_rust` stack (#49525, #49820, #49834, #49821)

## Problem

### The bridge carries knowledge it does not need

`python-bridge/src/redis/store/*.rs` was meant to be pure PyO3 conversion, but each family file also spelled store knowledge by hand. `rate_limits.rs` matched `"requests"`, `"tokens"`, `"window"` and `"max_parallel_requests"`, `router_usage.rs` matched `"tpm"` and `"rpm"`, `router_cooldowns.rs` named every `CooldownCacheValue` field, and `spend_counters.rs` and `auth_cache.rs` each mapped Redis client errors to kind strings. Those strings are Redis key suffixes and value shapes that `redis-store` already owns and spelled a second time

### Every store picks its own request phase

Whether an op goes out alone, rides the request's pre-call batch, or rides its post-call batch is a request concern, yet each store decides it inside its own method names

| Store | How it picks | Methods |
|---|---|---|
| `router_cooldowns` | always pre-call | `declare_active` |
| `spend_counters` | phase in the name | `declare_read_many_pre_call`, `increment_many_post_call` |
| `auth_cache` | phase in the name | `set_pre_call`, `delete_pre_call`, `declare_get_many` |
| `rate_limits` | `Option<&RequestRedis>` plus a private `Route`, and separate `*_post_call` methods | 8 script methods, 3 post-call methods |

The Rust stores call `request.pre(client)` or `request.post(client)` in 13 places. The Python wrappers in `litellm/redis/stores/*.py` repeat it one layer up, each looking up `active_request_redis_batches()`, checking for Redis and a request, picking `native()` or `native_post_call()`, and wrapping the result in `native_result`, with `rate_limits.py` adding its own `_declare` and `_post_call_request`. Moving an op between phases means a new method in Rust, the bridge, the stub and Python, and the names disagree across stores

`Route` (`Batch` or `Direct`) is shared infrastructure but lives in `rate_limits/scripts.rs`, so no other store can send an op alone, and `Pending<T>` there carries a second `Reply::Declared | Reply::Direct` enum just to resolve a reply

A pre-call read has to be awaited through `NativeBatchResult` so the native flush hooks run first and late readers such as the routing prefetch join the same trip. Every store has to remember that, and a bare `pending.resolve()` silently drops them

### Commands are written three times in redis-client

`commands.rs` builds each `redis::Cmd` once, but the surfaces around it repeat every command

| Surface | What it repeats |
|---|---|
| `RedisClient` (`client.rs`) | an async method per command applying the namespace, querying and decoding |
| `Pipeline` (`pipeline.rs`) | a builder per command applying the namespace and wrapping an `Op { entry, key, idempotent }` |
| `RedisBatch` (`batch.rs`) | a `BatchOp` variant per command, a match in `fn pipeline()` mapping it to `Pipeline`, and a match in `run_alone` mapping it to `RedisClient` |

One new batchable command touches five places

## Layers

| Layer | Owns | Does not know about |
|---|---|---|
| `redis-client` | commands as values, reply decoding, `RedisClient`, `Pipeline`, `RedisBatch` and its alone mode | requests, phases, keyspaces |
| request layer, `RequestRedis` and `Send` (in `redis-store` for now) | a request's pre and post batches, the post-call deadline, `spawn`, the shutdown drain, and turning a phase plus a client into a batch | keyspaces, Python |
| store modules (`router_cooldowns`, `router_usage`, `spend_counters`, `auth_cache`, `rate_limits`) | keys, value codecs, memory tiers, scripts, post-call follow-ups, Python value shapes through `FromValue` and `IntoValue` | which phase they run in |
| `python-bridge` | PyO3 conversion, `NativePending`, `NativeSend`, exception types | store logic |
| Python (`litellm/caching/redis_batch.py`, `litellm/redis/stores/*.py`) | `Phase`, `NativeTarget`, coexistence with stores still on redis-py | Rust internals |

`RequestRedis` depends only on `redis-client` and a Tokio handle, so it could move out of `redis-store`. It stays out of `redis-client`, whose `AGENTS.md` keeps it free of domain knowledge, and a request with pre and post phases is a host lifecycle concept. Keep it in `redis-store`, which both hosts already depend on, and move it to its own crate only when something besides the stores needs request batches

## redis-client

### One command value

```rust
pub struct Command<T> {
    plan: Plan,
    key: String,
    idempotent: bool,
    decode: fn(Value) -> Result<T, Error>,
}

enum Plan {
    Commands(Vec<Cmd>),
    Script(ScriptCall),
    IncrKeepingTtl { amount: Amount, ttl: Duration },
}
```

Each command is one constructor in `commands.rs`

```rust
pub fn get<T: FromRedisValue>(key: &str) -> Command<Option<T>>;
pub fn mget<T: FromRedisValue>(keys: &[&str]) -> Command<Vec<Option<T>>>;
pub fn set(key: &str, value: impl ToSingleRedisArg, ttl: Duration) -> Command<()>;
pub fn del(keys: &[&str]) -> Command<usize>;
pub fn expire(key: &str, ttl: Duration) -> Command<bool>;
pub fn incr_by_float(key: &str, amount: Amount, expiry: Expiry) -> Command<f64>;
pub fn eval<T: FromRedisValue>(script: &Script, keys: &[&str], args: ...) -> Command<T>;
```

The constructor arguments are the wire encode, through redis-rs `ToRedisArgs` (`Amount` writes an `int` with `str()` and a `float` with `repr()`, like redis-py). `decode` is the matching reply decode. Both stay generic, so `Command` has no separate `encode`

### Composing a store codec

```rust
impl<T> Command<T> {
    pub fn map<U>(self, decode: fn(T) -> Result<U, Error>) -> Command<U>;
}
```

A store attaches its keyspace codec without a new type, and the same `Command<Cooldown>` works on every executor

```rust
commands::get::<Vec<u8>>(&key).map(Cooldown::decode)
batch.add(commands::set(&key, cooldown.encode(), ttl))
```

### Three executors, one entry point each

```rust
impl RedisClient {
    pub async fn send<T>(&self, command: Command<T>) -> Result<T, Error>;
}
impl Pipeline {
    pub fn add<T>(self, command: Command<T>) -> Self;
}
impl RedisBatch {
    pub fn add<T>(&self, command: Command<T>) -> Declared<T>;
}
```

The executor applies its client's namespace to `key` once. `BatchOp`, `batch.rs`'s `fn pipeline()`, `run_alone` and `Pipeline`'s private `Op` go away, and a new command is one constructor. Call sites change from `client.get::<String>(key).await` to `client.send(commands::get::<String>(key)).await` and from `batch.mget(&keys)` to `batch.add(commands::mget(&keys))`. `subscribe` returns a stream and stays a method

### Alone versus pipelined behavior is explicit

A command run alone follows the `RedisCache` method behind it, and a pipelined one follows Python's pipeline helper, which is why `Plan` keeps the two cases that differ

| Command | Alone | Pipelined |
|---|---|---|
| `incr_by_float` with `Expiry::KeepExisting` | `INCRBYFLOAT`, read `TTL`, `EXPIRE` only a key with none, like `async_increment` | lowered to `Refresh`, `INCRBYFLOAT` plus `EXPIRE`, like `RedisBatch.increment` |
| `eval` | `EVALSHA`, and on `NOSCRIPT` a `SCRIPT LOAD` then `EVALSHA`, like redis-py's `Script` (commit `91ddc9f032f`) | `EVALSHA` with the pipeline's retry |

Each command with both forms gets a contract test against the commands Redis receives, so the forms never drift from Python's

### A batch that runs every op alone

```rust
impl RedisBatch {
    pub fn alone(client: RedisClient) -> Self;
}
```

`flush` already runs each op alone on a cluster. The alone flag takes that path everywhere, so a fresh alone batch is "send it now" and the first `resolve()` sends it. `Declared<T>` becomes the one reply handle for every route, which removes `rate_limits::scripts`'s `Route` and `Reply` enums and reduces its `Pending<T>` to nothing once `Command::map` carries the decode

## Request layer

```rust
pub enum Send<'a> {
    Now,
    PreCall(&'a RequestRedis),
    PostCall(&'a RequestRedis),
}

impl Send<'_> {
    pub fn batch(&self, client: &RedisClient) -> RedisBatch {
        match self {
            Self::Now => RedisBatch::alone(client.clone()),
            Self::PreCall(request) => request.pre(client),
            Self::PostCall(request) => request.post(client),
        }
    }
}
```

The caller decides the phase, because only it knows when it needs the answer. The store owns its client, which is optional and can differ per store, so the caller passes `Send` and the store calls `send.batch(&client)`, the one place the request and the client meet

`RequestRedis` keeps its current API, `pre`, `post`, `spawn`, `flush_pre` and `flush_post`. The pre-call batch goes out at the first `resolve()` or at `flush_pre()`. The post-call batch goes out at `flush_post()` or `post_call_deadline` after its first declaration, and `flush_post` waits for the follow-ups spawned on it so the shutdown drain keeps fallback writes in flight

## Stores

### Each op exists once

```rust
impl RateLimits {
    pub fn token_increment(&self, send: Send<'_>, increments: &[Increment]) -> Result<Declared<Vec<f64>>, Error> {
        let batch = send.batch(self.redis.as_ref().ok_or(Error::NoRedis)?);
        Ok(scripts::token_increment(&batch, increments))
    }
}
impl RouterCooldowns {
    pub fn active(&self, send: Send<'_>, model_ids: &[String]) -> Result<Declared<Vec<(String, Cooldown)>>, Error>;
}
impl SpendCounters {
    pub fn read_many(&self, send: Send<'_>, counters: &[SpendCounter]) -> Declared<Option<Vec<Option<Number>>>>;
}
impl AuthCache {
    pub fn get_many(&self, send: Send<'_>, keys: &[String]) -> Declared<Option<Vec<Option<Value>>>>;
}
```

### Post-call follow-ups stay with the store

Nobody resolves a post-call write, so work after it settles runs on `request.spawn`. A post-call method exists only where that follow-up exists, as a thin wrapper over the shared op

```rust
pub fn token_increment_post_call(self: &Arc<Self>, request: &RequestRedis, increments: Vec<Increment>) -> Result<(), Error> {
    let declared = self.token_increment(Send::PostCall(request), &increments)?;
    let store = Arc::clone(self);
    request.spawn(async move {
        if declared.settled().await.is_err() {
            let _ = store.increment_many(&increments).await;
        }
    });
    Ok(())
}
```

A plain post-call write is `op(Send::PostCall(request), ...)` with the `Declared` dropped

### Spellings and shapes live with the store

Key suffixes and wire spellings derive from strum on the store's enums, per `litellm-rust/.agents/skills/rust-string-enums/SKILL.md`, so there is one spelling per value

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::EnumString, strum::IntoStaticStr)]
pub enum Kind {
    #[strum(serialize = "requests")]
    Requests,
    #[strum(serialize = "tokens")]
    Tokens,
}
```

The same goes for `router_usage::Kind` (`tpm`, `rpm`), `spend_counters::Reading` (`redis`, `memory`), and `ErrorKind` (`breaker_open`, `timeout`, `transport`, `rejected`, `decode`, `encode`, `memory`), with `Error::kind()` on each store error and `From<&litellm_redis_client::Error> for ErrorKind`. `rate_limits::Entry::new(suffix, scope)` reads the suffix Python names an entry by, and the window and gauge suffixes are constants shared with the key builders

### Python value shapes

When the Python side of a store type is plain data, the store type converts through `litellm_python_compat::Value`, so the bridge never names a field

```rust
pub trait FromValue: Sized {
    fn from_value(value: &Value) -> Result<Self, DecodeError>;
}
pub trait IntoValue {
    fn into_value(self) -> Value;
}
```

`Cooldown::to_value` and `from_value` are the first case, and `encode` and `decode` are built on them (`repr(&self.to_value())` and `Self::from_value(&python_value(raw)?)`). `Counter`, `Scope`, `Entry`, `Increment` and `AtomicCounter` implement the same pair for the tuples Python sends, so each family's Python wire shape is a `redis-store` contract that parity tests can pin without pyo3. Unlike a store trait, this pair has a generic consumer, the bridge's `Py<T>`

Going through `Value` costs one small intermediate allocation over extracting a tuple directly, a few times per request at most, which readability outranks here

## python-bridge

### What stays

The bridge keeps only what pyo3 and the orphan rule force on it

1. `NativePending`, the one resolve-once handle for a declared op
2. `NativeSend`, the Python face of `Send`
3. `Py<T>` over `FromValue` and `IntoValue`, plus `PyValue`, `PyNumber` (the int versus float check) and the few non-data shapes such as `PyLocal`
4. Reply views such as the rate limiter's `(over_limit, applied)` tuples, which are Python-specific
5. `NativeRedisStoreError(kind, message)`, built once by `kind_error(error.kind(), &error)`
6. The `redis_async` and `redis_sync` helpers, which attach the `ServiceLogging` labels and run one scoped op
7. Host settings projection (`policy.rs`)

Each pyclass keeps its own explicit `#[pymethods]` list, with no store trait and no macro writing methods

### NativePending

```rust
#[pyclass(frozen, name = "NativePending")]
pub(crate) struct NativePending {
    resolve: Mutex<Option<Resolve>>,
    call_type: &'static str,
}
```

`resolve()` takes the future once, flushing the op's batch if it has not gone out, and raises `RuntimeError` on a second call. `call_type` is the `ServiceLogging` label, attached at resolve time where the traffic happens. Rust futures are lazy, so building the future at declare time sends nothing. It replaced `NativeDeclaredCooldowns`, `NativeDeclaredSpendCounters`, `NativeDeclaredAuthReads`, `NativeDeclaredAuthWrite` and `NativeRateLimitPending`

### NativeSend

`redis-store` stays free of pyo3, so the enum mirroring `Send` lives in the bridge. Each variant carries exactly what it needs, so there are no phase strings and no phase without a request

```rust
#[pyclass(frozen, name = "NativeSend")]
pub(crate) enum NativeSend {
    Now(),
    PreCall(Py<NativeRequestRedis>),
    PostCall(Py<NativeRequestRedis>),
}

impl NativeSend {
    fn send<'a>(&'a self, py: Python<'a>) -> Send<'a> {
        match self {
            Self::Now() => Send::Now,
            Self::PreCall(request) => Send::PreCall(&request.get().0),
            Self::PostCall(request) => Send::PostCall(&request.get().0),
        }
    }
}
```

`NativeRequestRedis` is `frozen`, so `Py::get()` borrows its `RequestRedis` without a clone. `Now()` needs parentheses because PyO3 complex enums do not mix unit and tuple variants. Every store method takes `send: &NativeSend` and returns `NativePending`

```rust
fn token_increment(&self, py: Python<'_>, increments_: Vec<Py<Increment>>, send: &NativeSend) -> PyResult<NativePending> {
    pending("token_increment_script", self.0.token_increment(send.send(py), &increments(increments_)), std::convert::identity)
}
```

Stub

```python
@final
class NativeSend:
    @final
    class Now(NativeSend):
        def __new__(cls) -> NativeSend.Now: ...
    @final
    class PreCall(NativeSend):
        def __new__(cls, request: NativeRequestRedis, /) -> NativeSend.PreCall: ...
    @final
    class PostCall(NativeSend):
        def __new__(cls, request: NativeRequestRedis, /) -> NativeSend.PostCall: ...
```

## Python

### Phase and NativeTarget in redis_batch.py

```python
class Phase(Enum):
    PRE_CALL = auto()
    POST_CALL = auto()


@dataclass(frozen=True, slots=True)
class NativeTarget:
    send: NativeSend
    batches: RequestRedisBatches | None

    def result(self, pending: NativePending[_T]) -> Awaitable[_T]:
        return self.batches.native_result(pending) if self.batches else pending.resolve()


def native_target(phase: Phase) -> NativeTarget:
    batches: Final = active_request_redis_batches()
    if batches is None:
        return NativeTarget(NativeSend.Now(), None)
    match phase:
        case Phase.PRE_CALL:
            return NativeTarget(NativeSend.PreCall(batches.native()), batches)
        case Phase.POST_CALL:
            return NativeTarget(NativeSend.PostCall(batches.native_post_call()), batches)
```

`result()` makes the flush-hook rule part of the API. A pre-call read always goes through `native_result`, which awaits `flush_native_pre()` and its hooks before `resolve()`, so no store can drop late readers. `native()` and `native_post_call()` return the same `NativeRequestRedis`, and `POST_CALL` keeps calling `native_post_call()` because it registers the request with the shutdown drain

### Python store ops

```python
def check_and_increment(self, counters: Sequence[AtomicCounter], target: NativeTarget) -> Awaitable[AtomicReply]:
    return _atomic_reply(target.result(self._handle.check_and_increment(tuple(map(_atomic, counters)), target.send)))
```

This drops the `declare_x`, `x` and `x_post_call` triples, the per-store `_declare` and `_post_call_request`, and the `request=` keyword

### App code

```python
target = native_target(Phase.PRE_CALL)
window, checked, spend, cooldowns = await asyncio.gather(
    rate_limits.sliding_window(counters, now, window_size, target),
    rate_limits.check_and_increment(atomic_counters, target),
    spend_counters.read_many(spend_keys, target),
    router_cooldowns.active(model_ids, target),
)
```

Inside a request the first await flushes, so the four reads share one trip. Outside one the target is `Now()` and each read goes alone, and the calling code is the same. Post-call writes keep their current call sites, such as `token_increment_post_call(increments)`, and pick `Phase.POST_CALL` inside the store

| | Pre-call | Post-call |
|---|---|---|
| For | reads the request needs | writes nobody waits on |
| Python gets | `NativePending` through `NativeTarget.result` | nothing it resolves |
| Sends on | the first `resolve()` or `flush_pre()` | `flush_post()` or the deadline |
| Waits on the reply | Python, through `resolve()` | Rust, through `settled()` in a spawned task |

## Coexisting with stores still on redis-py

`RequestRedisBatches` already flushes both kinds of batch each phase, the Python `RedisBatch` per backend (`batch()` and `post_call()`) and the one `NativeRequestRedis` (`native()`), gathering both in `flush_all` and `flush_post_call`. `native_target` only touches the native half. A store on redis-py keeps `batch(redis_cache)`, and a store that migrates switches its Python wrapper from `batch()` to `native_target()`, so nothing outside `litellm/redis/stores/` changes and stores move one at a time

Until the last store on a Redis moves, a request touching both kinds sends two concurrent pipelines per phase to it, which is what already happens today. The two pipelines have no order between them, which is fine because no store reads another's keys within a batch

## Not doing

1. A `RedisStore` trait. Nothing consumes stores generically, and their ops, inputs and replies all differ. What they share is values, `Send`, `Declared<T>` and `Command<T>`
2. A `Sink` with its own command methods. It would be a fourth copy of every command, and `RedisBatch::alone` gives the same routing with none
3. Python resolving post-call writes. They are fire-and-forget by design and `flush_post` already waits for their follow-ups at shutdown
4. Keyspace codecs on `Command`. `redis-client` stays domain-agnostic, and `Command::map` composes the store's codec instead
5. Typed exception kinds for router stores and rate limits, which raise `RuntimeError` today. That is a behavior change to decide on its own

## Open questions

1. `declare_*` returns `None` today when there is no Redis, and callers fall back to their own read. With `Now()` always available, the proposal is that each op answers from memory without Redis, the way `RateLimits::get` does, which changes every `declare_*` caller
2. `Declared` from an alone batch needs its own way to run, since `Declared::resolve` currently flushes its shared batch. A boxed future polled on first await should do, but it is not prototyped
3. The three `*_post_call` rate limit methods return an already finished awaitable around synchronous work, and could return `None` instead
4. Rate limit parameters `counters_`, `scopes_` and `increments_` leak their trailing underscore into Python keyword names, and renaming the helpers fixes it

## Order of work

1. `refactor(redis-client)` below the store stack: `Command<T>`, `Command::map`, `send` and `add` on the three executors, `RedisBatch::alone`, and the contract tests for commands with both forms
2. #49525: `Send` beside `RequestRedis`, `FromValue` and `IntoValue`, the bridge's `NativeSend` and `Py<T>`, Python's `Phase`, `NativeTarget` and `native_target`, and router cooldowns and usage converted
3. #49820: spend counters converted
4. #49834: auth cache converted
5. #49821: rate limits converted, dropping `_declare`, `_post_call_request` and the `request=` keyword

Each step is a new commit on its branch, restacked bottom up. Behavior stays the same except that ops limited to pre-call today can also go alone, and the existing store tests plus one `rstest` case per `Send` variant pin it

## Already done in the stack

1. `store/mod.rs` gained `redis_async`, `redis_sync`, `NativePending` and `ttl` (#49525), and each store method makes one helper call
2. `NativeRedisStoreError(kind, message)` replaced `NativeSpendCounterError` and `NativeAuthCacheError` (#49820, #49834)
3. `rate_limits::Kind` and `Entry::new`, `router_usage::Kind`, `Cooldown::to_value` and `from_value`, `spend_counters::Reading`, and `ErrorKind` with `Error::kind()` moved the spellings and shapes into `redis-store`, with `PyValue` moved down to #49525 so cooldowns can use it. Those last commits were restacked locally and not yet pushed when this was written
