# PR sketches

Interface sketches for each milestone in `migration-plan.md`. They show the shape each PR brings, not finished code. Bodies are elided with `...` where they are mechanical. Names and signatures here are what reviewers should hold a PR to, and any change to them updates this file in the same PR

## 1. Transport, store layer and response polling

Python only. Accepted behavior changes are listed in `migration-plan.md`

### `litellm/redis/transport/contract.py`

```python
Arg: TypeAlias = str | bytes | int | float
EvalReply: TypeAlias = int | bytes | None | tuple["EvalReply", ...]


class Admit(Protocol):
    def __call__(self, call: Callable[[], Awaitable[_T]], /) -> Awaitable[_T]: ...


@dataclass(frozen=True, slots=True)
class RedisFailure:
    kind: Literal["transport", "server", "decode", "breaker_open"]
    message: str
    timeout: bool = False


def ttl_ms(ttl: timedelta) -> int:
    return max(1, math.ceil(ttl / timedelta(milliseconds=1)))


class RedisTransport(Protocol):
    async def get(self, key: str) -> bytes | None | RedisFailure: ...

    async def mget(self, keys: Sequence[str]) -> tuple[bytes | None, ...] | RedisFailure: ...

    async def set(self, key: str, value: Arg, *, ttl: timedelta | None, nx: bool = False) -> bool | RedisFailure: ...

    async def delete(self, *keys: str) -> int | RedisFailure: ...

    async def eval(self, script: str, keys: Sequence[str], args: Sequence[Arg]) -> EvalReply | RedisFailure: ...
```

### `litellm/redis/transport/breaker.py`

```python
async def run_under_breaker(
    breaker: RedisCircuitBreaker, name: str, admit: Admit, call: Callable[[], Awaitable[_T]]
) -> _T | RedisFailure:
    try:
        return await _run_under_circuit_breaker(breaker, name, lambda: admit(call))
    except RedisCircuitBreakerOpenError as e:
        return RedisFailure(kind="breaker_open", message=str(e))
    except Exception as e:
        failure: Final = _redis_failure(e)
        if failure is None:
            raise
        return failure


def _redis_failure(exc: BaseException) -> RedisFailure | None:
    if _is_redis_health_failure(exc):
        return RedisFailure(kind="transport", message=str(exc), timeout=is_redis_timeout_failure(exc))
    if isinstance(exc, RedisError):
        return RedisFailure(kind="server", message=str(exc))
    return None
```

`_run_under_circuit_breaker`, `_is_redis_health_failure`, `is_redis_timeout_failure` and `RedisCircuitBreakerOpenError` are the ones in `litellm/caching/redis_cache.py` today, imported, not copied. That keeps the admission record, the generation check, the swallowed-failure check and the timeout class on the breaker identical for transport and legacy traffic. `admit` is the cache's `_admit_async_call`, which on `RedisClusterCache` runs the call through the cluster connection gate. Anything that is not a redis-py error propagates, since it is a bug and not a Redis outcome

### `litellm/redis/transport/events.py`

```python
@dataclass(frozen=True, slots=True)
class RedisServiceEvents:
    service_logging: ServiceLogging

    async def observe(self, call_type: str, call: Callable[[], Awaitable[_T | RedisFailure]]) -> _T | RedisFailure:
        if not self.service_logging.has_sinks():
            result: Final = await call()
            _log_failure(call_type, result)
            return result
        ...
```

`has_sinks()` is new on `ServiceLogging` and is `bool(litellm.service_callback)`. The hooks are awaited through `ServiceLogging.emit_guarded`, so a callback that raises never fails the Redis call. `...` times the call, logs a failure through the timeout throttle with the caller's `with_service_target`, and emits one success or failure REDIS event in the caller's task. No event for `breaker_open`

### `litellm/redis/transport/redis_py.py`

The only module outside `litellm/caching/` that imports redis-py

```python
@dataclass(frozen=True, slots=True)
class RedisPyTransport:
    client: Callable[[], AsyncRedisCommands]
    namespace: str | None
    breaker: RedisCircuitBreaker
    admit: Admit
    events: RedisServiceEvents

    async def get(self, key: str) -> bytes | None | RedisFailure:
        return _bytes_reply(await self._run("get", lambda: self.client().get(self._key(key))))

    async def set(self, key: str, value: Arg, *, ttl: timedelta | None, nx: bool = False) -> bool | RedisFailure:
        px: Final = None if ttl is None else ttl_ms(ttl)
        return _bool_reply(await self._run("set", lambda: self.client().set(self._key(key), value, px=px, nx=nx)))

    async def eval(self, script: str, keys: Sequence[str], args: Sequence[Arg]) -> EvalReply | RedisFailure:
        sha: Final = _sha1(script)
        namespaced: Final = tuple(self._key(k) for k in keys)
        return _eval_reply(await self._run("eval", lambda: _evalsha_or_eval(self.client(), script, sha, namespaced, args)))

    def _key(self, key: str) -> str:
        if self.namespace is None or key.startswith(f"{self.namespace}:"):
            return key
        return f"{self.namespace}:{key}"

    async def _run(self, name: str, call: Callable[[], Awaitable[_T]]) -> _T | RedisFailure:
        return await self.events.observe(name, lambda: run_under_breaker(self.breaker, name, self.admit, call))
```

`_evalsha_or_eval` sends `EVALSHA` and retries with `EVAL` on `NoScriptError`, so there is no registration and no per-loop state. `_key` is `check_and_fix_namespace` moved, same rule

### `litellm/caching/redis_cache.py`

```python
class RedisCache(BaseCache):
    def transport(self) -> RedisTransport:
        return RedisPyTransport(
            client=self.init_async_client,
            namespace=self.namespace,
            breaker=self._circuit_breaker,
            admit=self._admit_async_call,
            events=RedisServiceEvents(self.service_logger_obj),
        )
```

`RedisClusterCache` inherits it and its `_admit_async_call` override carries the connection gate onto the transport

### `litellm/redis/stores/response_polling.py`

```python
KEY_PREFIX: Final = "litellm:polling:response:"


class ResponsePollingStore(Protocol):
    async def put(self, polling_id: str, state: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure: ...

    async def get(self, polling_id: str) -> Mapping[str, JsonValue] | None | RedisFailure: ...

    async def delete(self, polling_id: str) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportResponsePollingStore:
    transport: RedisTransport

    async def put(self, polling_id: str, state: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure:
        written: Final = await self.transport.set(_key(polling_id), json.dumps(json.dumps(state)), ttl=ttl)
        return written if isinstance(written, RedisFailure) else None
    ...
```

The double `json.dumps` is the existing wire format, kept byte for byte, and `get` undoes both. A reply that is not valid JSON twice over is a `decode` failure

### Caller

```python
store: Final = TransportResponsePollingStore(redis_cache.transport())
match await store.get(polling_id):
    case RedisFailure():
        return None
    case None:
        return None
    case state:
        return state
```

The caller keeps its own policy (here, fail open on any failure) and drops its timing, service hooks and health logging

### Tests

`tests/unit/redis/transport/test_redis_py.py` covers namespace applied once, TTL landing (`pttl`), `server` on a rejected command, `transport` on `server.connected = False`, `breaker_open` without touching the socket, the cluster gate being entered on a `RedisClusterCache`, and `eval` surviving `SCRIPT FLUSH`. Store tests go through the caller on `fake_redis_cache(server)` and seed with the old writer

## 2. Every remaining raw redis-py keyspace behind a store

Python only. The transport grows only the commands these keyspaces need (`incrby`, `expire`, `rpush`, `lpop`, `llen` as they come up), each with its fakeredis test. Before adding one, check whether the caller's sequence is really one `eval`

### Store per keyspace

One module each in `litellm/redis/stores/`

```text
mcp_single_use.py        McpSingleUseStore       claim, peek
pod_lock.py              PodLockStore            acquire, release
rate_limit.py            RateLimitStore          check_and_increment, decrement, read_windows
session_budget.py        SessionBudgetStore      add_spend, spend
iteration_count.py       IterationCountStore     increment
enqueued_tokens.py       EnqueuedTokensStore     add, total
login_throttle.py        LoginThrottleStore      record_failure, failures, clear
affinity.py              AffinityStore           pin, pinned
mcp_dcr.py               McpDcrStore             put_client, get_client
mcp_credential_refresh.py McpRefreshLockStore    acquire, release
```

Operation names are a starting point, each PR names them for what its caller wants. The spend update buffer is not here, it reaches Redis through `RedisCache` list pipelines, which stay

### Example, `litellm/redis/stores/mcp_single_use.py`

```python
_CLAIM: Final = """
local count = redis.call("incrbyfloat", KEYS[1], ARGV[1])
if redis.call("ttl", KEYS[1]) < 0 then
    redis.call("expire", KEYS[1], ARGV[2])
end
return count
"""


class McpSingleUseStore(Protocol):
    async def claim(self, key: str, *, ttl: timedelta) -> int | RedisFailure: ...

    async def peek(self, key: str) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpSingleUseStore:
    transport: RedisTransport

    async def claim(self, key: str, *, ttl: timedelta) -> int | RedisFailure:
        reply: Final = await self.transport.eval(_CLAIM, (key,), (1, ttl // timedelta(seconds=1)))
        return reply if isinstance(reply, RedisFailure) else _count(reply)

    async def peek(self, key: str) -> bool | RedisFailure:
        reply: Final = await self.transport.get(key)
        return reply if isinstance(reply, RedisFailure) else reply is not None
```

`claim` is what `async_increment(key, 1, ttl=...)` does today, `INCRBYFLOAT` then `EXPIRE` only when the key has no TTL, in one script instead of a pipeline plus a conditional third round trip. The value stays an `INCRBYFLOAT` string, so `_count` parses a float and returns its integer, and a non-numeric reply is a `decode` failure. The caller keeps `"first"` as count 1, `"replayed"` otherwise, and fails closed on any `RedisFailure`. `peek` and `claim` move in the same commit, since they are one keyspace

### Example, `litellm/redis/stores/pod_lock.py`

```python
_RELEASE: Final = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def lock_key(cronjob_id: str) -> str:
    return f"cronjob_lock:{cronjob_id}"


@dataclass(frozen=True, slots=True)
class Acquired:
    pass


@dataclass(frozen=True, slots=True)
class HeldBy:
    owner: str


class PodLockStore(Protocol):
    async def acquire(self, cronjob_id: str, owner: str, *, ttl: timedelta) -> Acquired | HeldBy | None | RedisFailure: ...

    async def release(self, cronjob_id: str, owner: str) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportPodLockStore:
    transport: RedisTransport

    async def acquire(self, cronjob_id: str, owner: str, *, ttl: timedelta) -> Acquired | HeldBy | None | RedisFailure:
        set_reply: Final = await self.transport.set(lock_key(cronjob_id), _encode(owner), ttl=ttl, nx=True)
        if isinstance(set_reply, RedisFailure):
            return set_reply
        if set_reply:
            return Acquired()
        held: Final = await self.transport.get(lock_key(cronjob_id))
        if held is None or isinstance(held, RedisFailure):
            return held
        return _decode(held)

    async def release(self, cronjob_id: str, owner: str) -> bool | RedisFailure:
        reply: Final = await self.transport.eval(_RELEASE, (lock_key(cronjob_id),), (_encode(owner),))
        return reply if isinstance(reply, RedisFailure) else reply == 1


def _encode(owner: str) -> str:
    return json.dumps(owner)


def _decode(raw: bytes) -> HeldBy | RedisFailure: ...
```

The value is `json.dumps(pod_id)`, the bytes `async_set_cache` writes today, and the TTL stays whole seconds from the caller. `None` from `acquire` is the lock expiring between the `SET NX` and the `GET`, which the caller already treats as not acquired

The caller keeps the policy. `allow_reentrant` decides whether `HeldBy(self.pod_id)` counts as acquired, any `RedisFailure` is `False`, and the acquired and released events stay in `PodLockManager`. The GET plus DEL fallback after a failed script goes, as accepted in `migration-plan.md`, unless the pre-merge check finds a supported Redis setup with scripting disabled, in which case it stays in `PodLockManager`

### Lint gate

```toml
[lint.flake8-tidy-imports.banned-api]
"redis".msg = "reach Redis through litellm.redis (see .agents/skills/redis-access)"
```

Allowed in `litellm/redis/transport/redis_py.py`, `litellm/caching/redis_cache.py` and `litellm/caching/redis_cluster_cache.py` (they import `redis.exceptions` for the breaker and own the generic verbs), plus a per-file ignore for each remaining redis-py user listed in `migration-plan.md`, so the list only shrinks

## 3. Rust client parity

Rust only, nothing calls the new paths yet. This is 49189, and its PR body is the source of truth. The surface milestone 4 relies on

```rust
// crates/redis-client
impl RedisClient {
    pub async fn connect(config: RedisConfig, events: Arc<dyn RedisEvents>) -> Result<Self, ConnectError>;
    pub async fn get<T: FromRedisValue>(&self, key: &str) -> Result<Option<T>, Error>;
    pub async fn set_nx(&self, key: &str, value: impl ToSingleRedisArg, ttl: Duration) -> Result<bool, Error>;
    pub async fn eval<T: FromRedisValue>(&self, script: &Script, keys: &[&str], args: impl ToRedisArgs) -> Result<T, Error>;
    pub fn breaker_state(&self) -> BreakerState;
}

pub enum Error { Transport(RedisError), Server(RedisError), Decode(RedisError), BreakerOpen }

// crates/redis-connect
pub struct RedisConfig { /* standalone, cluster, sentinel, tls, credential source, namespace, timeouts */ }
```

`Error` maps one to one onto `RedisFailure.kind`, and keys get the namespace inside the client exactly like the transport

## 4. Bridge and the first native store

### `litellm-rust/crates/redis-store/src/pod_lock.rs`

```rust
const RELEASE: &str = r#"
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"#;

pub fn lock_key(cronjob_id: &str) -> String {
    format!("cronjob_lock:{cronjob_id}")
}

pub enum Acquire {
    Acquired,
    HeldBy(String),
    Expired,
}

pub async fn acquire(client: &RedisClient, cronjob_id: &str, owner: &str, ttl: Duration) -> Result<Acquire, Error>;

pub async fn release(client: &RedisClient, cronjob_id: &str, owner: &str) -> Result<bool, Error>;

fn encode(owner: &str) -> String {
    python_compat::json::dumps(&Value::Str(owner.into())).expect("a string always encodes")
}
```

Same keys, script text and codec as the Python store. The codec goes through `python-compat` so it matches `json.dumps` including ASCII escaping

### `litellm-rust/crates/python-bridge/src/redis/`

```rust
// client.rs, one per process, shared by every native store
#[pyclass(frozen)]
pub struct NativeRedisClient(Arc<RedisClient>);

#[pymethods]
impl NativeRedisClient {
    #[staticmethod]
    fn connect(settings: RedisSettings) -> PyResult<Self>;
}

// pod_lock.rs
#[pyclass(frozen)]
pub struct NativePodLockHandle(Arc<RedisClient>);

#[pymethods]
impl NativePodLockHandle {
    #[new]
    fn new(client: &NativeRedisClient) -> Self;
    fn acquire<'py>(&self, py: Python<'py>, cronjob_id: String, owner: String, ttl_ms: u64) -> PyResult<Bound<'py, PyAny>>;
    fn release<'py>(&self, py: Python<'py>, cronjob_id: String, owner: String) -> PyResult<Bound<'py, PyAny>>;
}
```

Each awaitable resolves to a generated frozen Pydantic result or a `RedisFailure`-shaped error value, never a raised exception for a Redis problem. The breaker lives in `RedisClient`, and its transitions reach `redis_breaker_state_counts()` through one host call in `litellm/rust_bridge/host/`

### `litellm/redis/stores/pod_lock.py`

```python
@dataclass(frozen=True, slots=True)
class NativePodLockStore:
    native: NativePodLockHandle
    events: RedisServiceEvents

    async def acquire(self, cronjob_id: str, owner: str, *, ttl: timedelta) -> Acquired | HeldBy | None | RedisFailure:
        return await self.events.observe("acquire", lambda: _acquire_reply(self.native.acquire(cronjob_id, owner, ttl_ms(ttl))))

    async def release(self, cronjob_id: str, owner: str) -> bool | RedisFailure:
        return await self.events.observe("release", lambda: _release_reply(self.native.release(cronjob_id, owner)))


def pod_lock_store(redis_cache: RedisCache, native: NativeRedisClient | None) -> PodLockStore:
    if native is None:
        return TransportPodLockStore(redis_cache.transport())
    return NativePodLockStore(NativePodLockHandle(native), RedisServiceEvents(redis_cache.service_logger_obj))
```

`native` is `None` when the Rust extension is off or the Redis settings use something the Rust client does not support yet. That selection happens once at construction and is the only branch. `PodLockManager` takes a `PodLockStore` instead of a `RedisCache`

### Tests

`tests/integration/redis/test_pod_lock_cross_impl.py` runs on one real Redis from `litellm-redis-testing`, standalone and cluster

```python
@pytest.mark.parametrize(("holder", "contender"), [(transport, native), (native, transport)])
async def test_lock_taken_by_one_stack_is_seen_and_released_only_by_its_owner(holder, contender): ...
```

It also checks that the TTL lands, that a Rust pod cannot release a Python pod's lock and the reverse, and that the owner releases across stacks. Rust side tests for `redis-store` run against the same Redis

### Caller switch

A separate commit or PR. The proxy builds `NativeRedisClient` once at startup and passes it to `pod_lock_store`. Proof is a live proxy pair, one pod on each stack, contending for the same cron lock
