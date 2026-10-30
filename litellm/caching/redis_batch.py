"""``RedisCache``-shaped operations on the transport's pipeline engine, and the request scope that hands each
backend one engine per request.

A ``RedisBatch`` collects MGETs, Lua scripts and increments declared by unrelated callers and sends them in one
round trip, together with any transport commands declared on the same engine. Every declaration returns an
awaitable; awaiting one flushes whatever has been declared so far, so callers keep their existing ``await`` shape
and their own error handling while sharing the wire. Redis Cluster clients run each operation on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from datetime import timedelta
from types import MappingProxyType, TracebackType
from typing import Final, Protocol

from litellm.caching.redis_cache import RedisCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.redis.transport.batch import (
    MIXED_PIPELINE_TARGET,
    BatchResult,
    PipelineBackend,
    PipelineBatch,
    SettledHook,
)
from litellm.redis.transport.client import RedisPipeline
from litellm.redis.transport.contract import Arg

__all__ = (
    "MIXED_PIPELINE_TARGET",
    "POST_CALL_FLUSH_DEADLINE_SECONDS",
    "BatchResult",
    "RedisBatch",
    "RegisteredScript",
    "RequestRedisBatches",
    "SettledHook",
    "active_post_call_redis_batch",
    "active_request_redis_batch",
    "active_request_redis_batches",
    "drain_post_call_redis_batches",
    "flush_post_call_redis_batches",
    "post_call_batch_engine",
    "request_batch_engine",
    "request_redis_batch_scope",
)

POST_CALL_FLUSH_DEADLINE_SECONDS: Final = 1.0


class RegisteredScript(Protocol):
    def __call__(self, keys: Sequence[str], args: Sequence[Arg]) -> Awaitable[object]: ...


class _MGet:
    __slots__ = ("_keys", "_redis_cache")

    def __init__(self, redis_cache: RedisCache, keys: Sequence[str]) -> None:
        self._redis_cache: Final = redis_cache
        self._keys: Final[tuple[str, ...]] = tuple(dict.fromkeys(keys))

    def enqueue(self, pipe: RedisPipeline) -> int:
        pipe.mget(tuple(self._redis_cache.check_and_fix_namespace(key=key) for key in self._keys))
        return 1

    def resolve(self, replies: Sequence[object]) -> Mapping[str, object]:
        values: Final = replies[0]
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"MGET reply is not a list: {type(values).__name__}")
        return MappingProxyType(
            {key: self._redis_cache._get_cache_logic(value) for key, value in zip(self._keys, values)}  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownArgumentType]  # shared decode with async_batch_get_cache
        )

    def recover(self, error: Exception) -> Mapping[str, object]:
        raise error

    async def run_alone(self) -> Mapping[str, object]:
        found: Mapping[str, object] = await self._redis_cache.async_batch_get_cache(key_list=list(self._keys))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped cache API
        if any(key not in found for key in self._keys):
            raise ConnectionError("batch get did not return every key")
        return found


class _Script:
    __slots__ = ("_args", "_keys", "_redis_cache", "_run", "_sha")

    def __init__(
        self,
        redis_cache: RedisCache,
        source: str,
        run: RegisteredScript,
        keys: Sequence[str],
        args: Sequence[Arg],
    ) -> None:
        self._redis_cache: Final = redis_cache
        self._sha: Final = hashlib.sha1(source.encode()).hexdigest()  # noqa: S324  # EVALSHA identifies scripts by SHA-1
        self._run: Final = run
        self._keys: Final[tuple[str, ...]] = tuple(keys)
        self._args: Final[tuple[Arg, ...]] = tuple(args)

    def enqueue(self, pipe: RedisPipeline) -> int:
        namespaced: Final = tuple(self._redis_cache.check_and_fix_namespace(key=key) for key in self._keys)
        pipe.evalsha(self._sha, len(namespaced), *namespaced, *self._args)
        return 1

    def resolve(self, replies: Sequence[object]) -> object:
        return replies[0]

    def recover(self, error: Exception) -> object:
        raise error

    async def run_alone(self) -> object:
        return await self._run(keys=self._keys, args=self._args)


class _Increment:
    __slots__ = ("_key", "_redis_cache", "_ttl", "_value")

    def __init__(self, redis_cache: RedisCache, key: str, value: float, ttl: int | None) -> None:
        self._redis_cache: Final = redis_cache
        self._key: Final = key
        self._value: Final = value
        self._ttl: Final = ttl

    def enqueue(self, pipe: RedisPipeline) -> int:
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

    def recover(self, error: Exception) -> float:
        raise error

    async def run_alone(self) -> float:
        value: object = await self._redis_cache.async_increment(key=self._key, value=self._value, ttl=self._ttl)  # pyright: ignore[reportUnknownMemberType]  # untyped cache API
        if not isinstance(value, (int, float)):
            raise TypeError(f"increment did not return a number: {type(value).__name__}")
        return float(value)


class _Set:
    """SET with the cache's TTL rules, same encoding as ``async_set_cache_pipeline_with_ttls``."""

    __slots__ = ("_key", "_redis_cache", "_ttl", "_value")

    def __init__(self, redis_cache: RedisCache, key: str, value: object, ttl: float | None) -> None:
        self._redis_cache: Final = redis_cache
        self._key: Final = key
        self._value: Final = value
        self._ttl: Final = ttl

    def enqueue(self, pipe: RedisPipeline) -> int:
        ttl: Final = self._redis_cache.get_ttl(ttl=self._ttl)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped cache API
        pipe.set(
            self._redis_cache.check_and_fix_namespace(key=self._key),
            json.dumps(self._value),
            ex=None if ttl is None else timedelta(seconds=ttl),
        )
        return 1

    def resolve(self, replies: Sequence[object]) -> None:
        return None

    def recover(self, error: Exception) -> None:
        raise error

    async def run_alone(self) -> None:
        await self._redis_cache.async_set_cache_pipeline_with_ttls(((self._key, self._value, self._ttl),))


class _Delete:
    """DEL of one key, the pipelined twin of ``async_delete_cache``."""

    __slots__ = ("_key", "_redis_cache")

    def __init__(self, redis_cache: RedisCache, key: str) -> None:
        self._redis_cache: Final = redis_cache
        self._key: Final = key

    def enqueue(self, pipe: RedisPipeline) -> int:
        pipe.delete(self._redis_cache.check_and_fix_namespace(key=self._key))
        return 1

    def resolve(self, replies: Sequence[object]) -> None:
        return None

    def recover(self, error: Exception) -> None:
        raise error

    async def run_alone(self) -> None:
        await self._redis_cache.async_delete_cache(self._key)


def pipeline_backend(redis_cache: RedisCache) -> PipelineBackend:
    return PipelineBackend(
        client=redis_cache.init_async_client,
        breaker=redis_cache._circuit_breaker,  # pyright: ignore[reportPrivateUsage]  # same breaker as the cache's own methods
        admit=redis_cache._admit_async_call,  # pyright: ignore[reportPrivateUsage]  # same connection gate as the cache's own methods
        service_logging=redis_cache.service_logger_obj,
        runs_alone=isinstance(redis_cache, RedisClusterCache),
    )


class RedisBatch:
    """Operations declared here go out in one pipeline the next time any of them is awaited or ``flush`` runs."""

    __slots__ = ("engine", "redis_cache")

    def __init__(self, redis_cache: RedisCache, name: str = "redis_batch") -> None:
        self.redis_cache: Final = redis_cache
        self.engine: Final = PipelineBatch(pipeline_backend(redis_cache), name=name)

    @property
    def name(self) -> str:
        return self.engine.name

    @property
    def flushes(self) -> int:
        return self.engine.flushes

    @property
    def pending(self) -> int:
        return self.engine.pending

    def mget(self, keys: Sequence[str]) -> BatchResult[Mapping[str, object]]:
        result: Final = self.engine.declare(_MGet(self.redis_cache, keys))
        result.on_settled(self._note_misses)
        return result

    def _note_misses(self, future: asyncio.Future[Mapping[str, object]]) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        self.engine.note_missing(key for key, value in future.result().items() if value is None)

    def read_as_missing(self, key: str) -> bool:
        """True when an MGET on this batch already found no value under ``key`` and nothing has set it since,
        so a per-key GET later in the same request can be answered without another round trip."""
        return self.engine.read_as_missing(key)

    def script(
        self, source: str, run: RegisteredScript, keys: Sequence[str], args: Sequence[Arg]
    ) -> BatchResult[object]:
        return self.engine.declare(_Script(self.redis_cache, source, run, keys, args))

    def increment(self, key: str, value: float, ttl: int | None = None) -> BatchResult[float]:
        return self.engine.declare(_Increment(self.redis_cache, key, value, ttl))

    def set(self, key: str, value: object, ttl: float | None = None) -> BatchResult[None]:
        self.engine.note_present(key)
        return self.engine.declare(_Set(self.redis_cache, key, value, ttl))

    def delete(self, key: str) -> BatchResult[None]:
        self.engine.note_missing((key,))
        return self.engine.declare(_Delete(self.redis_cache, key))

    def add_flush_hook(self, hook: Callable[[], None]) -> None:
        """Called at the start of every flush so lazily bound readers can declare their keys into the same trip."""
        self.engine.add_flush_hook(hook)

    async def flush(self) -> None:
        await self.engine.flush()


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
        self._batches: Final[dict[object, RedisBatch]] = {}  # mutable-ok: lazily filled per backend
        self._post_call: Final[dict[object, RedisBatch]] = {}  # mutable-ok: lazily filled per backend
        self.post_call_deadline: Final = post_call_deadline
        self._deadline: asyncio.TimerHandle | None = None
        self._deadline_flush: asyncio.Task[None] | None = None
        # Reads declared early for a consumer that runs later in the request, keyed by consumer name.
        self.prefetched: Final[dict[str, object]] = {}  # mutable-ok: armed pre-admission, taken at use

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


def request_batch_engine(redis_cache: RedisCache) -> PipelineBatch | None:
    """The engine the transport declares request-scoped commands on, shared with this backend's ``RedisBatch``."""
    batch: Final = active_request_redis_batch(redis_cache)
    return None if batch is None else batch.engine


def post_call_batch_engine(redis_cache: RedisCache) -> PipelineBatch | None:
    batch: Final = active_post_call_redis_batch(redis_cache)
    return None if batch is None else batch.engine
