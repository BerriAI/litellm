from __future__ import annotations

import asyncio
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from opentelemetry.trace import Span

from litellm._internal_context import service_target
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import BatchResult, active_request_redis_batches

_PREFETCH_SLOT: Final = "routing_read"


@dataclass(frozen=True, slots=True)
class TierRead:
    cache: DualCache
    keys: tuple[str, ...]


def selector_read(cache: DualCache, keys: Sequence[str]) -> TierRead:
    return TierRead(cache, tuple(keys))


def routing_target(reads: Sequence[TierRead]) -> str:
    if len(reads) < 2 or not reads[1].keys:
        return "router_cooldowns"
    return "router_cooldowns_usage" if reads[0].keys else "router_usage"


async def _backfill(read: TierRead, values: Mapping[str, object]) -> None:
    pending: Final = await read.cache._prepare_batch_get(  # pyright: ignore[reportPrivateUsage]  # store owns tier
        list(read.keys), local_only=True
    )
    missed: Final = {
        key: values[key]
        for key, local in zip(read.keys, pending.result)
        if local is None and values.get(key) is not None
    }
    await read.cache._apply_batch_get(pending, missed)  # pyright: ignore[reportPrivateUsage]  # store owns tier


@dataclass(frozen=True, slots=True)
class ArmedRead:
    keys: frozenset[str]
    fetched: frozenset[str]
    result: BatchResult[Mapping[str, object]]
    reservations: tuple[tuple[TierRead, Mapping[str, float | None]], ...]

    def release(self) -> None:
        for read, previous in self.reservations:
            read.cache._rollback_redis_batch_key_reservations(  # pyright: ignore[reportPrivateUsage]  # store owns tier
                dict(previous)
            )

    async def settle(self, future: asyncio.Future[Mapping[str, object]]) -> None:
        if future.cancelled() or future.exception() is not None:
            self.release()
            return
        try:
            for read, _ in self.reservations:
                await _backfill(read, future.result())
        except Exception:
            self.release()
            raise


def arm(reads: Sequence[TierRead]) -> None:
    request: Final = active_request_redis_batches()
    redis: Final = reads[0].cache.redis_cache if reads else None
    if request is None or redis is None or _PREFETCH_SLOT in request.prefetched:
        return
    reserved: Final = tuple((read, read.cache.reserve_redis_batch_reads(read.keys)) for read in reads)
    due_reads: Final = tuple(TierRead(read.cache, tuple(due)) for read, (due, _) in reserved)
    due: Final = tuple(itertools.chain.from_iterable(read.keys for read in due_reads))
    if not due:
        return
    with service_target(routing_target(due_reads)):
        result: Final = request.batch(redis).mget(due)
    prefetch: Final = ArmedRead(
        keys=frozenset(itertools.chain.from_iterable(read.keys for read in reads)),
        fetched=frozenset(due),
        result=result,
        reservations=tuple((read, reservation[1]) for read, (_, reservation) in zip(due_reads, reserved)),
    )
    result.on_settled(prefetch.settle)
    request.prefetched[_PREFETCH_SLOT] = prefetch


def armed() -> bool:
    request: Final = active_request_redis_batches()
    return request is not None and _PREFETCH_SLOT in request.prefetched


def _take(needed: Sequence[str]) -> ArmedRead | None:
    request: Final = active_request_redis_batches()
    if request is None:
        return None
    taken: Final = request.prefetched.pop(_PREFETCH_SLOT, None)
    if isinstance(taken, ArmedRead) and taken.keys.issuperset(needed):
        return taken
    if isinstance(taken, ArmedRead):
        taken.release()
    return None


async def _read_prefetched(reads: Sequence[TierRead]) -> Sequence[Sequence[object | None] | None] | None:
    prefetch: Final = _take(tuple(itertools.chain.from_iterable(read.keys for read in reads)))
    if prefetch is None:
        return None
    try:
        values: Final = await prefetch.result
    except Exception as e:  # noqa: BLE001  # shared read applies the tiers' existing Redis fallback
        verbose_router_logger.debug("routing prefetch failed, reading again: %s", e)
        return None
    pending_reads: Final = tuple(
        [
            await read.cache._prepare_batch_get(  # pyright: ignore[reportPrivateUsage]  # store owns tier
                list(read.keys), local_only=True
            )
            for read in reads
        ]
    )
    if any(
        any(key not in prefetch.fetched for key, value in zip(read.keys, pending.result) if value is None)
        for read, pending in zip(reads, pending_reads)
    ):
        return None
    return [
        await read.cache._apply_batch_get(  # pyright: ignore[reportPrivateUsage]  # store owns tier
            pending, {key: values.get(key) for key, value in zip(read.keys, pending.result) if value is None}
        )
        for read, pending in zip(reads, pending_reads)
    ]


async def read(reads: Sequence[TierRead], parent_otel_span: Span | None) -> Sequence[Sequence[object | None] | None]:
    with service_target(routing_target(reads)):
        prefetched: Final = await _read_prefetched(reads)
        if prefetched is not None:
            return prefetched
        return await DualCache.async_batch_get_cache_shared(
            tuple((read.cache, list(read.keys)) for read in reads), parent_otel_span=parent_otel_span
        )
