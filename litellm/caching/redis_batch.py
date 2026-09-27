"""Commands declared against one RedisCache, executed later as a single pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeVar, cast

from litellm.caching.redis_cache import (
    RedisCircuitBreakerOpenError,
    _enter_circuit_breaker,  # pyright: ignore[reportPrivateUsage]  # breaker admission shared with RedisCache
    _exit_circuit_breaker,  # pyright: ignore[reportPrivateUsage]  # breaker admission shared with RedisCache
    _record_swallowed_redis_failure,  # pyright: ignore[reportPrivateUsage]  # failure accounting shared with RedisCache
)
from litellm.types.services import ServiceTypes

if TYPE_CHECKING:
    from redis.asyncio.client import Pipeline
    from redis.commands.core import AsyncScript

    from litellm.caching.redis_cache import RedisCache

_CommandResult = TypeVar("_CommandResult")


@dataclass(frozen=True, slots=True)
class _IssuedCommand:
    """How many pipeline result slots a command occupies and how to decode its slice."""

    slot_count: int
    decode: Callable[[Sequence[object]], object]


if TYPE_CHECKING:
    _IssueFn = Callable[[Pipeline, bool], _IssuedCommand]
    _Declaration = tuple[asyncio.Future[object], _IssueFn]


def _fail_all(declarations: Sequence[_Declaration], exc: BaseException) -> None:
    for future, _ in declarations:
        if not future.done():
            future.set_exception(exc)


def _consume_unretrieved_exception(future: asyncio.Future[object]) -> None:
    """A declared future that never gets awaited must not warn at GC time."""
    if not future.cancelled():
        future.exception()


def _decode_value(redis_cache: RedisCache, value: object) -> object:
    return redis_cache._get_cache_logic(value)  # pyright: ignore[reportPrivateUsage]  # shared cache decode


class RedisBatch:
    """A single-use set of Redis commands that run together in one ``pipeline(transaction=False)``.

    Each declaring method returns an ``asyncio.Future`` that is resolved from the pipeline's
    results when ``execute()`` runs, or fails with the pipeline's own error. Keys are
    namespaced and values serialized/decoded exactly as ``RedisCache`` does itself.
    """

    def __init__(self, redis_cache: RedisCache) -> None:
        self._redis_cache: Final = redis_cache
        self._pending: list[_Declaration] = []
        self._executed = False

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    @property
    def command_count(self) -> int:
        return len(self._pending)

    def _declare(self, issue: _IssueFn) -> asyncio.Future[_CommandResult]:
        if self._executed:
            raise RuntimeError("RedisBatch is single-use: no commands can be declared after execute()")
        future: asyncio.Future[_CommandResult] = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_unretrieved_exception)
        declared: Final = cast(asyncio.Future[object], future)  # cast-ok: Future is invariant over the stored type
        self._pending.append((declared, issue))
        return future

    def mget(self, keys: Sequence[str]) -> asyncio.Future[dict[str, object | None]]:
        redis_cache: Final = self._redis_cache
        un_namespaced: Final = tuple(key for key in keys if key is not None)
        namespaced: Final = tuple(redis_cache.check_and_fix_namespace(key=key) for key in un_namespaced)

        def issue(pipe: Pipeline, is_cluster: bool) -> _IssuedCommand:
            if not namespaced:
                slot_count = 0
            elif is_cluster:
                for key in namespaced:
                    pipe.get(key)
                slot_count = len(namespaced)
            else:
                pipe.mget(namespaced)
                slot_count = 1

            def decode(items: Sequence[object]) -> dict[str, object | None]:
                values: Final = items if is_cluster else items[0] if items else []
                return {
                    (key.decode("utf-8") if isinstance(key, bytes) else key): _decode_value(redis_cache, value)
                    for key, value in zip(un_namespaced, values)
                }

            return _IssuedCommand(slot_count=slot_count, decode=decode)

        return self._declare(issue)

    def get(self, key: str) -> asyncio.Future[object | None]:
        redis_cache: Final = self._redis_cache
        namespaced: Final = redis_cache.check_and_fix_namespace(key=key)

        def issue(pipe: Pipeline, is_cluster: bool) -> _IssuedCommand:
            pipe.get(namespaced)
            return _IssuedCommand(
                slot_count=1,
                decode=lambda items: _decode_value(redis_cache, items[0] if items else None),
            )

        return self._declare(issue)

    def evalsha(
        self, script: AsyncScript, keys: Sequence[str], args: Sequence[int | float | str]
    ) -> asyncio.Future[object]:
        namespaced: Final = tuple(self._redis_cache.check_and_fix_namespace(key=key) for key in keys)

        def issue(pipe: Pipeline, is_cluster: bool) -> _IssuedCommand:
            scripts: Final = getattr(pipe, "scripts", None)
            if scripts is not None:
                scripts.add(script)
            pipe.evalsha(script.sha, len(namespaced), *namespaced, *args)
            return _IssuedCommand(slot_count=1, decode=lambda items: items[0] if items else None)

        return self._declare(issue)

    def incrbyfloat(self, key: str, amount: float, ttl: int | None = None) -> asyncio.Future[float]:
        namespaced: Final = self._redis_cache.check_and_fix_namespace(key=key)

        def issue(pipe: Pipeline, is_cluster: bool) -> _IssuedCommand:
            pipe.incrbyfloat(namespaced, amount)
            if ttl is not None:
                pipe.expire(namespaced, ttl)
            return _IssuedCommand(
                slot_count=1 + (1 if ttl is not None else 0),
                decode=lambda items: float(items[0]) if items else 0.0,
            )

        return self._declare(issue)

    def set(self, key: str, value: object, ttl: int | None = None) -> asyncio.Future[bool]:
        namespaced: Final = self._redis_cache.check_and_fix_namespace(key=key)
        serialized: Final = json.dumps(value)

        def issue(pipe: Pipeline, is_cluster: bool) -> _IssuedCommand:
            pipe.set(name=namespaced, value=serialized, ex=ttl)
            return _IssuedCommand(slot_count=1, decode=lambda items: bool(items[0]) if items else False)

        return self._declare(issue)

    def delete(self, key: str) -> asyncio.Future[int]:
        namespaced: Final = self._redis_cache.check_and_fix_namespace(key=key)

        def issue(pipe: Pipeline, is_cluster: bool) -> _IssuedCommand:
            pipe.delete(namespaced)
            return _IssuedCommand(slot_count=1, decode=lambda items: int(items[0]) if items else 0)

        return self._declare(issue)

    async def execute(self) -> None:
        self._executed = True
        declarations: Final = self._pending
        self._pending = []
        if not declarations:
            return
        redis_cache: Final = self._redis_cache
        try:
            admission = _enter_circuit_breaker(
                redis_cache._circuit_breaker,
                "redis_batch",  # pyright: ignore[reportPrivateUsage]  # same breaker async_batch_get_cache consults
            )
        except RedisCircuitBreakerOpenError as e:
            _fail_all(declarations, e)
            return
        start_time: Final = time.time()
        try:
            from redis.asyncio import RedisCluster  # deferred: redis is an optional dependency

            client: Final = redis_cache.init_async_client()
            is_cluster: Final = isinstance(client, RedisCluster)
            pipe: Final = client.pipeline(transaction=False)
            issued: Final = [issue(pipe, is_cluster) for _, issue in declarations]
            results: Final = await pipe.execute(raise_on_error=False)
        except BaseException as e:
            _fail_all(declarations, e)
            if not isinstance(e, Exception):  # cancellation and KeyboardInterrupt still fail every future
                raise
            _record_swallowed_redis_failure(
                redis_cache._circuit_breaker,
                e,  # pyright: ignore[reportPrivateUsage]  # same breaker async_batch_get_cache feeds
            )
            asyncio.create_task(
                redis_cache.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=time.time() - start_time,
                    error=e,
                    call_type="redis_batch.execute",
                    start_time=start_time,
                    end_time=time.time(),
                )
            )
            return
        position = 0
        for (future, _), command in zip(declarations, issued):
            items = results[position : position + command.slot_count]
            position += command.slot_count
            if future.done():
                continue
            error = next((item for item in items if isinstance(item, Exception)), None)
            if error is not None:
                future.set_exception(error)
                continue
            try:
                future.set_result(command.decode(items))
            except Exception as e:
                future.set_exception(e)
        _exit_circuit_breaker(
            redis_cache._circuit_breaker,
            admission,  # pyright: ignore[reportPrivateUsage]  # same breaker async_batch_get_cache consults
        )
        asyncio.create_task(
            redis_cache.service_logger_obj.async_service_success_hook(
                service=ServiceTypes.REDIS,
                duration=time.time() - start_time,
                call_type="redis_batch.execute",
                start_time=start_time,
                end_time=time.time(),
            )
        )
