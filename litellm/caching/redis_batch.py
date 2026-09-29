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
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType, TracebackType
from typing import Final, Generic, Protocol, TypeVar

from litellm._logging import verbose_logger
from litellm.caching.redis_cache import (
    RedisCache,
    _run_under_circuit_breaker,  # pyright: ignore[reportPrivateUsage]  # same health signal as every RedisCache method
    log_redis_failure,
)
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.types.services import ServiceTypes

_T = TypeVar("_T")
_ScriptArg = str | bytes | int | float


class RegisteredScript(Protocol):
    def __call__(self, keys: Sequence[str], args: Sequence[_ScriptArg]) -> Awaitable[object]: ...


class _RedisPipeline(Protocol):
    def mget(self, keys: Sequence[str]) -> object: ...
    def evalsha(self, sha: str, numkeys: int, *keys_and_args: _ScriptArg) -> object: ...
    def incrbyfloat(self, name: str, amount: float) -> object: ...
    def expire(self, name: str, time: timedelta) -> object: ...
    def set(self, name: str, value: str, ex: timedelta | None = None) -> object: ...
    async def execute(self, raise_on_error: bool = True) -> list[object]: ...


class _Op(Generic[_T]):
    """One declared operation: how many pipeline replies it consumes, how to turn them into a result, and
    how to run on its own when the batch cannot pipeline (cluster client, or a reply the pipeline cannot
    settle, like NOSCRIPT)."""

    __slots__ = ("future",)

    def __init__(self) -> None:
        self.future: Final[asyncio.Future[_T]] = asyncio.get_running_loop().create_future()
        self.future.add_done_callback(_mark_retrieved)

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
        found: Mapping[str, object] = await self._redis_cache.async_batch_get_cache(key_list=list(self._keys))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped cache API  # mutable-ok: the cache API takes a list
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


@dataclass(slots=True)
class RedisBatch:
    """Operations declared here go out in one pipeline the next time any of them is awaited or ``flush`` runs."""

    redis_cache: RedisCache
    name: str = "redis_batch"
    _pending: list[_Op[object]] = field(default_factory=list)  # mutable-ok: drained by flush
    _flush_hooks: list[Callable[[], None]] = field(default_factory=list)  # mutable-ok: append-only registry
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    flushes: int = 0

    def mget(self, keys: Sequence[str]) -> BatchResult[Mapping[str, object]]:
        return self._declare(_MGet(self.redis_cache, keys))

    def script(
        self, source: str, run: RegisteredScript, keys: Sequence[str], args: Sequence[_ScriptArg]
    ) -> BatchResult[object]:
        return self._declare(_Script(self.redis_cache, source, run, keys, args))

    def increment(self, key: str, value: float, ttl: int | None = None) -> BatchResult[float]:
        return self._declare(_Increment(self.redis_cache, key, value, ttl))

    def set(self, key: str, value: object, ttl: float | None = None) -> BatchResult[None]:
        return self._declare(_Set(self.redis_cache, key, value, ttl))

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

    async def _flush_pipeline(self, ops: Sequence[_Op[object]]) -> None:
        start_time: Final = time.time()
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
            asyncio.create_task(
                self.redis_cache.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=time.time() - start_time,
                    error=e,
                    call_type=f"{self.name}[{len(ops)}]",
                    start_time=start_time,
                    end_time=time.time(),
                )
            )
            for op in ops:
                op.future.set_exception(e)
            return
        asyncio.create_task(
            self.redis_cache.service_logger_obj.async_service_success_hook(
                service=ServiceTypes.REDIS,
                duration=time.time() - start_time,
                call_type=f"{self.name}[{len(ops)}]",
                start_time=start_time,
                end_time=time.time(),
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


class RequestRedisBatches:
    """One ``RedisBatch`` per Redis backend for the current request, so readers of different caches that
    share a server (the proxy's and the router's) share the pipeline."""

    __slots__ = ("_batches", "prefetched")

    def __init__(self) -> None:
        self._batches: Final[dict[object, RedisBatch]] = {}  # mutable-ok: lazily filled per backend
        # Reads declared early for a consumer that runs later in the request, keyed by consumer name.
        self.prefetched: Final[dict[str, object]] = {}  # mutable-ok: armed pre-admission, taken at use

    def batch(self, redis_cache: RedisCache) -> RedisBatch:
        key: Final = _backend_key(redis_cache)
        batch = self._batches.get(key)
        if batch is None:
            batch = RedisBatch(redis_cache, name="request_redis_batch")
            self._batches[key] = batch
        return batch

    async def flush_all(self) -> None:
        """Send whatever is still declared (write-backs nobody awaits) before the request scope closes."""
        await asyncio.gather(*(batch.flush() for batch in self._batches.values() if batch.pending))

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


class request_redis_batch_scope:
    """Redis reads declared inside share one pipeline per backend; nested scopes join the outer one."""

    __slots__ = ("_token",)

    def __init__(self) -> None:
        self._token: Token[RequestRedisBatches | None] | None = None

    def __enter__(self) -> RequestRedisBatches:
        outer: Final = _active_request_batches.get()
        if outer is not None:
            return outer
        batches: Final = RequestRedisBatches()
        self._token = _active_request_batches.set(batches)
        return batches

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._token is not None:
            _active_request_batches.reset(self._token)
