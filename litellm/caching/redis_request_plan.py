"""A request-scoped set of lazily executed Redis pipelines.

Declarations accumulate in one ``RedisBatch`` per ``RedisCache``. Awaiting any declared
future runs one round: every pending batch executes concurrently, then fresh batches open
for whatever gets declared next. A scope left with unexecuted declarations is drained by
``flush()`` (the ASGI middleware calls it at request exit).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, TypeVar

from litellm._logging import verbose_logger
from litellm.caching.redis_batch import RedisBatch
from litellm.caching.redis_cache import RedisCache

T = TypeVar("T")


class RedisRequestPlan:
    """Rounds of lazily executed pipelines: declarations accumulate per RedisCache, awaiting
    any future runs one pipeline per RedisCache for everything declared so far."""

    def __init__(self) -> None:
        self._open: dict[int, tuple[RedisCache, RedisBatch]] = {}
        self._lock: Final = asyncio.Lock()
        self._rounds = 0

    def batch_for(self, redis_cache: RedisCache) -> RedisBatch:
        entry: Final = self._open.get(id(redis_cache))
        if entry is not None and entry[0] is redis_cache:
            return entry[1]
        batch: Final = RedisBatch(redis_cache)
        self._open[id(redis_cache)] = (redis_cache, batch)
        return batch

    @property
    def rounds(self) -> int:
        return self._rounds

    async def resolve(self, future: asyncio.Future[T]) -> T:
        if not future.done():
            async with self._lock:
                if not future.done():
                    await self._execute_round()
        return await future

    async def flush(self) -> None:
        async with self._lock:
            await self._execute_round()

    async def _execute_round(self) -> None:
        entries: Final = self._open
        self._open = {}
        batches: Final = tuple(batch for _, batch in entries.values() if batch.pending)
        if not batches:
            return
        self._rounds += 1
        verbose_logger.debug(
            "redis_request_plan: round %d, %d commands",
            self._rounds,
            sum(batch.command_count for batch in batches),
        )
        await asyncio.gather(*(batch.execute() for batch in batches))


_active_plan: Final[ContextVar[RedisRequestPlan | None]] = ContextVar("redis_request_plan", default=None)


def active_redis_request_plan() -> RedisRequestPlan | None:
    return _active_plan.get()


@contextmanager
def redis_request_plan_scope() -> Iterator[RedisRequestPlan]:
    """Opens a plan for the current context; joins the existing one when already inside a scope."""
    existing: Final = _active_plan.get()
    if existing is not None:
        yield existing
        return
    plan: Final = RedisRequestPlan()
    token: Final = _active_plan.set(plan)
    try:
        yield plan
    finally:
        _active_plan.reset(token)
