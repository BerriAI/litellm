"""One Redis pipeline for several independent operations, each with its own result and its own failure.

A ``RedisBatch`` collects MGETs, Lua scripts and increments declared by unrelated callers and sends them
in one ``pipeline(transaction=False)`` round trip. Every declaration returns an awaitable; awaiting one
flushes whatever has been declared so far, so callers keep their existing ``await`` shape and their own
error handling while sharing the wire. Redis Cluster clients run each operation on its own, as before:
a cluster pipeline is per node anyway and the existing per-operation paths already group by slot.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import weakref
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType, TracebackType
from typing import Final, Generic, Protocol, TypeVar

from litellm._internal_context import (
    REDIS_FAMILIES_METADATA_KEY,
    current_service_target,
    service_caller,
    service_target,
)
from litellm._logging import verbose_logger
from litellm.caching.redis_cache import (
    RedisCache,
    _get_call_stack_info,  # pyright: ignore[reportPrivateUsage]  # same caller chain every RedisCache method reports
    _run_under_circuit_breaker,  # pyright: ignore[reportPrivateUsage]  # same health signal as every RedisCache method
    log_redis_failure,
)
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.types.services import ServiceTypes

_T = TypeVar("_T")
_ScriptArg = str | bytes | int | float
SettledHook = Callable[[asyncio.Future[_T]], Awaitable[None] | None]
POST_CALL_FLUSH_DEADLINE_SECONDS: Final = 1.0


class RegisteredScript(Protocol):
    def __call__(self, keys: Sequence[str], args: Sequence[_ScriptArg]) -> Awaitable[object]: ...


class _RedisPipeline(Protocol):
    def mget(self, keys: Sequence[str]) -> object: ...
    def evalsha(self, sha: str, numkeys: int, *keys_and_args: _ScriptArg) -> object: ...
    def incrbyfloat(self, name: str, amount: float) -> object: ...
    def expire(self, name: str, time: timedelta) -> object: ...
    def set(self, name: str, value: str, ex: timedelta | None = None) -> object: ...
    def delete(self, *names: str) -> object: ...
    async def execute(self, raise_on_error: bool = True) -> list[object]: ...


class _Op(Generic[_T]):
    """One declared operation: how many pipeline replies it consumes, how to turn them into a result, and
    how to run on its own when the batch cannot pipeline (cluster client, or a reply the pipeline cannot
    settle, like NOSCRIPT)."""

    __slots__ = ("caller", "future", "settled_hooks", "target")

    def __init__(self) -> None:
        self.future: Final[asyncio.Future[_T]] = asyncio.get_running_loop().create_future()
        self.future.add_done_callback(_mark_retrieved)
        self.settled_hooks: Final[list[SettledHook[_T]]] = []  # mutable-ok: append-only registry
        self.target: Final = current_service_target()
        self.caller: Final = _get_call_stack_info()

    async def run_settled_hooks(self) -> None:
        for hook in self.settled_hooks:
            await self._run_settled_hook(hook)

    async def _run_settled_hook(self, hook: SettledHook[_T]) -> None:
        try:
            follow_up: Final = hook(self.future)
            if follow_up is not None:
                await follow_up
        except Exception as e:  # noqa: BLE001  # one owner's follow-up must not stop the others
            verbose_logger.warning("redis batch settled hook failed: %s", e)

    def enqueue(self, pipe: _RedisPipeline) -> int:
        raise NotImplementedError

    def resolve(self, replies: Sequence[object]) -> _T:
        raise NotImplementedError

    async def run_alone(self) -> _T:
        raise NotImplementedError

    def settle(self, replies: Sequence[object]) -> Awaitable[None] | None:
        """Resolve from pipeline replies; return a coroutine when the op has to be retried on its own."""
        failure: Final = next((reply for reply in replies if isinstance(reply, Exception)), None)
        if failure is None:
            try:
                self.future.set_result(self.resolve(replies))
            except Exception as e:  # noqa: BLE001  # a reply this op cannot decode fails this op alone
                self.future.set_exception(e)
            return None
        if _is_missing_script(failure):
            return self._settle_alone()
        self.future.set_exception(failure)
        return None

    async def _settle_alone(self) -> None:
        try:
            with service_target(self.target), service_caller(self.caller):
                self.future.set_result(await self.run_alone())
        except Exception as e:  # noqa: BLE001  # the declaring caller owns the failure of its own operation
            self.future.set_exception(e)


def _is_missing_script(failure: Exception) -> bool:
    """Imported lazily: this module is reachable from a base ``import litellm`` while redis is not a base dependency."""
    from redis.exceptions import NoScriptError

    return isinstance(failure, NoScriptError)


def _mark_retrieved(future: asyncio.Future[object]) -> None:
    """A caller that stops awaiting (cancelled request) must not leave an 'exception never retrieved' log."""
    if not future.cancelled():
        future.exception()


class _MGet(_Op[Mapping[str, object]]):
    __slots__ = ("_keys", "_redis_cache")

    def __init__(self, redis_cache: RedisCache, keys: Sequence[str]) -> None:
        super().__init__()
        self._redis_cache: Final = redis_cache
        self._keys: Final[tuple[str, ...]] = tuple(dict.fromkeys(keys))

    def enqueue(self, pipe: _RedisPipeline) -> int:
        pipe.mget(tuple(self._redis_cache.check_and_fix_namespace(key=key) for key in self._keys))
        return 1

    def resolve(self, replies: Sequence[object]) -> Mapping[str, object]:
        values: Final = replies[0]
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"MGET reply is not a list: {type(values).__name__}")
        return MappingProxyType(
            {key: self._redis_cache._get_cache_logic(value) for key, value in zip(self._keys, values)}  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownArgumentType]  # shared decode with async_batch_get_cache
        )

    async def run_alone(self) -> Mapping[str, object]:
        found: Mapping[str, object] = await self._redis_cache.async_batch_get_cache(key_list=list(self._keys))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped cache API
        if any(key not in found for key in self._keys):
            raise ConnectionError("batch get did not return every key")
        return found


class _Script(_Op[object]):
    __slots__ = ("_args", "_keys", "_redis_cache", "_run", "_sha")

    def __init__(
        self,
        redis_cache: RedisCache,
        source: str,
        run: RegisteredScript,
        keys: Sequence[str],
        args: Sequence[_ScriptArg],
    ) -> None:
        super().__init__()
        self._redis_cache: Final = redis_cache
        self._sha: Final = hashlib.sha1(source.encode()).hexdigest()  # noqa: S324  # EVALSHA identifies scripts by SHA-1
        self._run: Final = run
        self._keys: Final[tuple[str, ...]] = tuple(keys)
        self._args: Final[tuple[_ScriptArg, ...]] = tuple(args)

    def enqueue(self, pipe: _RedisPipeline) -> int:
        namespaced: Final = tuple(self._redis_cache.check_and_fix_namespace(key=key) for key in self._keys)
        pipe.evalsha(self._sha, len(namespaced), *namespaced, *self._args)
        return 1

    def resolve(self, replies: Sequence[object]) -> object:
        return replies[0]

    async def run_alone(self) -> object:
        return await self._run(keys=self._keys, args=self._args)


class _Increment(_Op[float]):
    __slots__ = ("_key", "_redis_cache", "_ttl", "_value")

    def __init__(self, redis_cache: RedisCache, key: str, value: float, ttl: int | None) -> None:
        super().__init__()
        self._redis_cache: Final = redis_cache
        self._key: Final = key
        self._value: Final = value
        self._ttl: Final = ttl

    def enqueue(self, pipe: _RedisPipeline) -> int:
        name: Final = self._redis_cache.check_and_fix_namespace(key=self._key)
        pipe.incrbyfloat(name, self._value)
        if self._ttl is None:
            return 1
        pipe.expire(name, timedelta(seconds=self._ttl))
        return 2

    def resolve(self, replies: Sequence[object]) -> float:
        reply: Final = replies[0]
        if not isinstance(reply, (int, float, str, bytes)):
            raise TypeError(f"INCRBYFLOAT reply is not numeric: {type(reply).__name__}")
        return float(reply)

    async def run_alone(self) -> float:
        value: object = await self._redis_cache.async_increment(key=self._key, value=self._value, ttl=self._ttl)  # pyright: ignore[reportUnknownMemberType]  # untyped cache API
        if not isinstance(value, (int, float)):
            raise TypeError(f"increment did not return a number: {type(value).__name__}")
        return float(value)


class _Set(_Op[None]):
    """SET with the cache's TTL rules, same encoding as ``async_set_cache_pipeline_with_ttls``."""

    __slots__ = ("_key", "_redis_cache", "_ttl", "_value")

    def __init__(self, redis_cache: RedisCache, key: str, value: object, ttl: float | None) -> None:
        super().__init__()
        self._redis_cache: Final = redis_cache
        self._key: Final = key
        self._value: Final = value
        self._ttl: Final = ttl

    def enqueue(self, pipe: _RedisPipeline) -> int:
        ttl: Final = self._redis_cache.get_ttl(ttl=self._ttl)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped cache API
        pipe.set(
            self._redis_cache.check_and_fix_namespace(key=self._key),
            json.dumps(self._value),
            ex=None if ttl is None else timedelta(seconds=ttl),
        )
        return 1

    def resolve(self, replies: Sequence[object]) -> None:
        return None

    async def run_alone(self) -> None:
        await self._redis_cache.async_set_cache_pipeline_with_ttls(((self._key, self._value, self._ttl),))


class _Delete(_Op[None]):
    """DEL of one key, the pipelined twin of ``async_delete_cache``."""

    __slots__ = ("_key", "_redis_cache")

    def __init__(self, redis_cache: RedisCache, key: str) -> None:
        super().__init__()
        self._redis_cache: Final = redis_cache
        self._key: Final = key

    def enqueue(self, pipe: _RedisPipeline) -> int:
        pipe.delete(self._redis_cache.check_and_fix_namespace(key=self._key))
        return 1

    def resolve(self, replies: Sequence[object]) -> None:
        return None

    async def run_alone(self) -> None:
        await self._redis_cache.async_delete_cache(self._key)


class BatchResult(Generic[_T]):
    """Awaitable handle for one declared operation; awaiting it flushes the batch it belongs to."""

    __slots__ = ("_batch", "_op")

    def __init__(self, batch: RedisBatch, op: _Op[_T]) -> None:
        self._batch: Final = batch
        self._op: Final = op

    def __await__(self) -> Generator[object, None, _T]:
        return self._wait().__await__()

    async def _wait(self) -> _T:
        if not self._op.future.done():
            await self._batch.flush()
        return self._op.future.result()

    @property
    def done(self) -> bool:
        return self._op.future.done()

    def on_settled(self, hook: SettledHook[_T]) -> None:
        """For an owner that does not await: runs inside the flush once this operation has its result or
        failure (or was cancelled with the pipeline), so the flush completes with the follow-up done."""
        self._op.settled_hooks.append(hook)


@dataclass(slots=True)
class RedisBatch:
    """Operations declared here go out in one pipeline the next time any of them is awaited or ``flush`` runs."""

    redis_cache: RedisCache
    name: str = "redis_batch"
    _pending: list[_Op[object]] = field(default_factory=list)  # mutable-ok: drained by flush
    _flush_hooks: list[Callable[[], None]] = field(default_factory=list)  # mutable-ok: append-only registry
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _misses: set[str] = field(default_factory=set)  # mutable-ok: keys an MGET of this request read as absent
    flushes: int = 0

    def mget(self, keys: Sequence[str]) -> BatchResult[Mapping[str, object]]:
        op: Final = _MGet(self.redis_cache, keys)
        op.future.add_done_callback(self._note_misses)
        return self._declare(op)

    def _note_misses(self, future: asyncio.Future[Mapping[str, object]]) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        self._misses.update(key for key, value in future.result().items() if value is None)

    def read_as_missing(self, key: str) -> bool:
        """True when an MGET on this batch already found no value under ``key`` and nothing has set it since,
        so a per-key GET later in the same request can be answered without another round trip."""
        return key in self._misses

    def script(
        self, source: str, run: RegisteredScript, keys: Sequence[str], args: Sequence[_ScriptArg]
    ) -> BatchResult[object]:
        return self._declare(_Script(self.redis_cache, source, run, keys, args))

    def increment(self, key: str, value: float, ttl: int | None = None) -> BatchResult[float]:
        return self._declare(_Increment(self.redis_cache, key, value, ttl))

    def set(self, key: str, value: object, ttl: float | None = None) -> BatchResult[None]:
        self._misses.discard(key)
        return self._declare(_Set(self.redis_cache, key, value, ttl))

    def delete(self, key: str) -> BatchResult[None]:
        self._misses.add(key)
        return self._declare(_Delete(self.redis_cache, key))

    def add_flush_hook(self, hook: Callable[[], None]) -> None:
        """Called at the start of every flush so lazily bound readers can declare their keys into the same trip."""
        self._flush_hooks.append(hook)

    @property
    def pending(self) -> int:
        return len(self._pending)

    def _declare(self, op: _Op[_T]) -> BatchResult[_T]:
        self._pending.append(op)  # pyright: ignore[reportArgumentType]  # heterogeneous ops share the flush loop
        return BatchResult(self, op)

    async def flush(self) -> None:
        async with self._lock:
            for hook in self._flush_hooks:
                hook()
            ops: Final = tuple(self._pending)
            self._pending.clear()
            if not ops:
                return
            self.flushes += 1
            try:
                if isinstance(self.redis_cache, RedisClusterCache):
                    await asyncio.gather(*(op._settle_alone() for op in ops))  # pyright: ignore[reportPrivateUsage]  # batch owns its ops
                else:
                    await self._flush_pipeline(ops)
            finally:
                for op in ops:
                    if not op.future.done():
                        op.future.cancel()
                await asyncio.gather(*(op.run_settled_hooks() for op in ops))

    async def _flush_pipeline(self, ops: Sequence[_Op[object]]) -> None:
        start_time: Final = time.time()
        target, metadata = _pipeline_service_event(ops)
        widths: list[int] = []  # mutable-ok: filled while enqueuing

        async def run() -> list[object]:
            client: Final = self.redis_cache.init_async_client()
            async with client.pipeline(transaction=False) as pipe:
                widths.extend(op.enqueue(pipe) for op in ops)
                return await pipe.execute(raise_on_error=False)

        try:
            replies: Final = await _run_under_circuit_breaker(self.redis_cache._circuit_breaker, self.name, run)  # pyright: ignore[reportPrivateUsage]  # same breaker as the cache's own methods
        except Exception as e:  # noqa: BLE001  # each declaring caller applies its own Redis fallback
            log_redis_failure(verbose_logger, logging.WARNING, f"{self.name}: pipeline of {len(ops)} ops failed", e)
            with service_target(target):
                asyncio.create_task(
                    self.redis_cache.service_logger_obj.async_service_failure_hook(
                        service=ServiceTypes.REDIS,
                        duration=time.time() - start_time,
                        error=e,
                        call_type=self.name,
                        start_time=start_time,
                        end_time=time.time(),
                        event_metadata=metadata,
                    )
                )
            for op in ops:
                op.future.set_exception(e)
            return
        with service_target(target):
            asyncio.create_task(
                self.redis_cache.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=time.time() - start_time,
                    call_type=self.name,
                    start_time=start_time,
                    end_time=time.time(),
                    event_metadata=metadata,
                )
            )
        retries: list[Awaitable[None]] = []  # mutable-ok: collected while slicing replies
        offset = 0
        for op, width in zip(ops, widths):
            retry = op.settle(replies[offset : offset + width])
            offset += width
            if retry is not None:
                retries.append(retry)
        if retries:
            await asyncio.gather(*retries)


MIXED_PIPELINE_TARGET: Final = "mixed"


def _pipeline_service_event(ops: Sequence[_Op[object]]) -> tuple[str | None, dict[str, int | str]]:
    """The target and metadata of one pipeline flush: the one key family every op was declared under, or
    ``"mixed"`` plus the sorted families when owners of several families share the trip."""
    families: Final = sorted({op.target for op in ops if op.target is not None})
    if len(families) > 1:
        return MIXED_PIPELINE_TARGET, {"op_count": len(ops), REDIS_FAMILIES_METADATA_KEY: ",".join(families)}
    return next(iter(families), None), {"op_count": len(ops)}


def _backend_key(redis_cache: RedisCache) -> object:
    """Two ``RedisCache`` instances built from the same connection settings and namespace talk to the same server
    under the same key prefix, so the proxy's cache and the router's cache share one pipeline (the router gets its
    port as a string, hence the ``str`` comparison); a cache whose settings cannot be compared (a test double) gets
    its own."""
    try:
        settings: Final = tuple(sorted((str(k), str(v)) for k, v in redis_cache.redis_kwargs.items() if v is not None))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]  # untyped cache API
    except AttributeError:
        return ("instance", id(redis_cache))
    return (type(redis_cache), redis_cache.namespace, settings)


_open_post_call: Final[weakref.WeakSet[RequestRedisBatches]] = weakref.WeakSet()
"""Requests whose post-call batch still holds declared ops, so a shutdown can send them before Redis goes away."""


class RequestRedisBatches:
    """One ``RedisBatch`` per Redis backend for the current request, so readers of different caches that
    share a server (the proxy's and the router's) share the pipeline.

    The post-call batches hold the writes nothing waits on (counters, token scripts, the response cache).
    They flush once, when the success or failure callbacks have all run, or at ``post_call_deadline``
    seconds after the first declaration when no callback phase closes them."""

    __slots__ = (
        "__weakref__",
        "_batches",
        "_deadline",
        "_deadline_flush",
        "_post_call",
        "post_call_deadline",
        "prefetched",
    )

    def __init__(self, post_call_deadline: float = POST_CALL_FLUSH_DEADLINE_SECONDS) -> None:
        self._batches: Final[dict[object, RedisBatch]] = {}
        self._post_call: Final[dict[object, RedisBatch]] = {}
        self.post_call_deadline: Final = post_call_deadline
        self._deadline: asyncio.TimerHandle | None = None
        self._deadline_flush: asyncio.Task[None] | None = None
        # Reads declared early for a consumer that runs later in the request, keyed by consumer name.
        self.prefetched: Final[dict[str, object]] = {}

    def batch(self, redis_cache: RedisCache) -> RedisBatch:
        key: Final = _backend_key(redis_cache)
        batch = self._batches.get(key)
        if batch is None:
            batch = RedisBatch(redis_cache, name="request_redis_batch")
            self._batches[key] = batch
        return batch

    def post_call(self, redis_cache: RedisCache) -> RedisBatch:
        key: Final = _backend_key(redis_cache)
        existing: Final = self._post_call.get(key)
        batch: Final = (
            existing
            if existing is not None
            else self._post_call.setdefault(key, RedisBatch(redis_cache, name="post_call_redis_batch"))
        )
        if self._deadline is None:
            self._deadline = asyncio.get_running_loop().call_later(self.post_call_deadline, self._flush_on_deadline)
        _open_post_call.add(self)
        return batch

    def _flush_on_deadline(self) -> None:
        self._deadline = None
        self._deadline_flush = asyncio.ensure_future(self.flush_post_call())

    async def flush_all(self) -> None:
        """Send whatever is still declared (write-backs nobody awaits) before the request scope closes."""
        await asyncio.gather(*(batch.flush() for batch in self._batches.values() if batch.pending))

    async def flush_post_call(self) -> None:
        """One pipeline per backend for the post-call writes; the deadline is disarmed since this is that flush."""
        if self._deadline is not None:
            self._deadline.cancel()
            self._deadline = None
        await asyncio.gather(*(batch.flush() for batch in self._post_call.values() if batch.pending))
        if not any(batch.pending for batch in self._post_call.values()):
            _open_post_call.discard(self)

    @property
    def batches(self) -> tuple[RedisBatch, ...]:
        return tuple(self._batches.values())


_active_request_batches: Final[ContextVar[RequestRedisBatches | None]] = ContextVar(
    "request_redis_batches", default=None
)


def active_request_redis_batch(redis_cache: RedisCache) -> RedisBatch | None:
    """The request's batch for this backend, or None outside a ``request_redis_batch_scope``."""
    batches: Final = _active_request_batches.get()
    if batches is None:
        return None
    return batches.batch(redis_cache)


def active_request_redis_batches() -> RequestRedisBatches | None:
    return _active_request_batches.get()


def active_post_call_redis_batch(redis_cache: RedisCache) -> RedisBatch | None:
    """The request's post-call batch for this backend, or None outside a ``request_redis_batch_scope``."""
    batches: Final = _active_request_batches.get()
    if batches is None:
        return None
    return batches.post_call(redis_cache)


async def flush_post_call_redis_batches() -> None:
    """Called where the success and failure callbacks of a request have all run."""
    batches: Final = _active_request_batches.get()
    if batches is not None:
        await batches.flush_post_call()


async def drain_post_call_redis_batches() -> None:
    """Sends every post-call batch still waiting on its callbacks or deadline; for the shutdown path."""
    await asyncio.gather(*(batches.flush_post_call() for batches in tuple(_open_post_call)))


class request_redis_batch_scope:
    """Redis reads declared inside share one pipeline per backend; nested scopes join the outer one."""

    __slots__ = ("_post_call_deadline", "_token")

    def __init__(self, post_call_deadline: float = POST_CALL_FLUSH_DEADLINE_SECONDS) -> None:
        self._token: Token[RequestRedisBatches | None] | None = None
        self._post_call_deadline: Final = post_call_deadline

    def __enter__(self) -> RequestRedisBatches:
        outer: Final = _active_request_batches.get()
        if outer is not None:
            return outer
        batches: Final = RequestRedisBatches(post_call_deadline=self._post_call_deadline)
        self._token = _active_request_batches.set(batches)
        return batches

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._token is not None:
            _active_request_batches.reset(self._token)
