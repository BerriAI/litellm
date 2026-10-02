"""
One Redis round trip for the reads a request needs before a deployment can be picked.

The cooldown filter (`CooldownCache`, its own `DualCache`) and usage-based selection
(`LowestTPMLoggingHandler_v2`, the router cache) each issue their own MGET because they live in
different objects. `RoutingReadBatch` fetches both key sets in one
`DualCache.async_batch_get_cache_shared` while the healthy deployments are being resolved and hands
the usage slice to the strategy, so selection does not read again.
"""

import asyncio
import itertools
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import BatchResult, active_request_redis_batches
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2, PrefetchedUsage
from litellm.router_utils.cooldown_cache import CooldownCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.router import Router


_PREFETCH_SLOT: Final = "routing_read"


async def _backfill_prefetched_cache(
    cache: DualCache,
    due_keys: tuple[str, ...],
    values: Mapping[str, object],
) -> None:
    cache_keys: Final = list(due_keys)
    prepare_batch_get: Final = cache._prepare_batch_get  # pyright: ignore[reportPrivateUsage]  # memory backfill
    pending: Final = await prepare_batch_get(cache_keys, local_only=True)
    redis_values: Final = {
        key: values[key]
        for key, local in zip(due_keys, pending.result)
        if local is None and values.get(key) is not None
    }
    apply_batch_get: Final = cache._apply_batch_get  # pyright: ignore[reportPrivateUsage]  # cache backfill
    await apply_batch_get(pending, redis_values)


@dataclass(frozen=True, slots=True)
class RoutingPrefetch:
    """The cooldown and usage keys of a model group, declared on the request's Redis batch before admission
    flushes it, so the routing read rides the same round trip as the rate limiter's Lua calls."""

    keys: frozenset[str]
    fetched: frozenset[str]
    result: BatchResult[Mapping[str, object]]
    reservations: tuple[tuple[DualCache, tuple[str, ...], dict[str, float | None]], ...]

    def release(self) -> None:
        for cache, _, previous_access_times in self.reservations:
            cache._rollback_redis_batch_key_reservations(  # pyright: ignore[reportPrivateUsage]  # rollback
                previous_access_times
            )

    async def _settle(self, future: asyncio.Future[Mapping[str, object]]) -> None:
        if future.cancelled():
            self.release()
            return
        if future.exception() is not None:
            self.release()
            return

        values: Final = future.result()
        try:
            for cache, due_keys, _ in self.reservations:
                await _backfill_prefetched_cache(cache, due_keys, values)
        except Exception:
            self.release()
            raise

    @staticmethod
    def arm(
        litellm_router_instance: "Router",
        usage_selector: LowestTPMLoggingHandler_v2 | None,
        deployments: list,
    ) -> None:
        request: Final = active_request_redis_batches()
        redis_cache: Final = litellm_router_instance.cache.redis_cache
        if request is None or redis_cache is None or _PREFETCH_SLOT in request.prefetched:
            return
        cooldown_keys: Final = tuple(
            CooldownCache.get_cooldown_cache_key(model_id) for model_id in litellm_router_instance.get_model_ids()
        )
        usage_keys: Final = (
            () if usage_selector is None else tuple(itertools.chain(*usage_selector.usage_counter_keys(deployments)))
        )
        keys: Final = (*cooldown_keys, *usage_keys)
        cooldown_store: Final = litellm_router_instance.cooldown_cache.cooldown_store
        cooldown_due, cooldown_previous = cooldown_store.reserve_redis_batch_reads(cooldown_keys)
        usage_cache: Final = None if usage_selector is None else usage_selector.router_cache
        usage_reservation: Final = None if usage_cache is None else usage_cache.reserve_redis_batch_reads(usage_keys)
        usage_due: Final = () if usage_reservation is None else tuple(usage_reservation[0])
        due: Final = (*cooldown_due, *usage_due)
        reservations: Final = (
            (cooldown_store, tuple(cooldown_due), cooldown_previous),
            *(
                ()
                if usage_cache is None or usage_reservation is None
                else ((usage_cache, usage_due, usage_reservation[1]),)
            ),
        )
        if not due:
            return
        result: Final = request.batch(redis_cache).mget(due)
        prefetch: Final = RoutingPrefetch(
            keys=frozenset(keys), fetched=frozenset(due), result=result, reservations=reservations
        )
        result.on_settled(prefetch._settle)
        request.prefetched[_PREFETCH_SLOT] = prefetch

    @staticmethod
    def armed() -> bool:
        request: Final = active_request_redis_batches()
        return request is not None and _PREFETCH_SLOT in request.prefetched

    @staticmethod
    def take(needed: Sequence[str]) -> "RoutingPrefetch | None":
        """The armed prefetch when it covers every key this read needs; taken once, so a retry reads fresh."""
        request: Final = active_request_redis_batches()
        if request is None:
            return None
        armed: Final = request.prefetched.pop(_PREFETCH_SLOT, None)
        if isinstance(armed, RoutingPrefetch) and armed.keys.issuperset(needed):
            return armed
        if isinstance(armed, RoutingPrefetch):
            armed.release()
        return None


_active_routing_read_batch: Final[ContextVar["RoutingReadBatch | None"]] = ContextVar(
    "routing_read_batch", default=None
)


class RoutingReadBatch:
    def __init__(self, usage_selector: LowestTPMLoggingHandler_v2 | None) -> None:
        self.usage_selector: Final = usage_selector
        self.prefetched_usage: PrefetchedUsage | None = None

    @staticmethod
    @contextmanager
    def scoped(batch: "RoutingReadBatch | None") -> Iterator[None]:
        token: Final = _active_routing_read_batch.set(batch)
        try:
            yield
        finally:
            _active_routing_read_batch.reset(token)

    @staticmethod
    def active() -> "RoutingReadBatch | None":
        return _active_routing_read_batch.get()

    @staticmethod
    def for_strategy(strategy: str | None, selector: object) -> "RoutingReadBatch | None":
        """Usage-based routing reads its counters with the cooldown state; every other strategy reads only the
        cooldown state, and only through this batch when the request armed a prefetch for it. Otherwise the
        router's plain cooldown read stays in charge."""
        if strategy == "usage-based-routing-v2" and isinstance(selector, LowestTPMLoggingHandler_v2):
            return RoutingReadBatch(usage_selector=selector)
        return RoutingReadBatch(usage_selector=None) if RoutingPrefetch.armed() else None

    async def async_get_cooldown_deployments(
        self,
        litellm_router_instance: "Router",
        healthy_deployments: list,
        parent_otel_span: "Span | None",
    ) -> list[str]:
        """
        `_async_get_cooldown_deployments`, with the strategy's tpm/rpm counters for
        `healthy_deployments` fetched in the same MGET and kept as `prefetched_usage`.
        """
        model_ids: Final = litellm_router_instance.get_model_ids()
        cooldown_keys: Final = [CooldownCache.get_cooldown_cache_key(model_id) for model_id in model_ids]
        selector: Final = self.usage_selector
        usage_keys: Final = (
            () if selector is None else tuple(itertools.chain(*selector.usage_counter_keys(healthy_deployments)))
        )
        reads: Final = (
            (litellm_router_instance.cooldown_cache.cooldown_store, cooldown_keys),
            *(() if selector is None else ((selector.router_cache, list(usage_keys)),)),
        )
        results: Final = await self._read_prefetched(reads) or await DualCache.async_batch_get_cache_shared(
            reads, parent_otel_span=parent_otel_span
        )
        cooldown_results: Final = results[0]
        if selector is not None:
            usage_values: Final = results[1]
            self.prefetched_usage = PrefetchedUsage(
                keys=frozenset(usage_keys),
                values=None if usage_values is None else MappingProxyType(dict(zip(usage_keys, usage_values))),
            )

        cooldown_models: Final = litellm_router_instance.cooldown_cache.active_cooldowns_from_results(
            model_ids, cooldown_results
        )
        verbose_router_logger.debug("retrieve cooldown models: %s", cooldown_models)
        return [model_id for model_id, _ in cooldown_models]

    @staticmethod
    async def _read_prefetched(
        reads: Sequence[tuple[DualCache, list[str]]],
    ) -> list[list[object | None] | None] | None:
        """Serve the reads from the request's armed `RoutingPrefetch`, backfilling each cache's memory tier as
        its own batch read would. None when nothing usable was armed or the prefetch failed."""
        prefetch: Final = RoutingPrefetch.take(tuple(itertools.chain.from_iterable(keys for _, keys in reads)))
        if prefetch is None:
            return None
        try:
            values: Final = await prefetch.result
        except Exception as e:  # noqa: BLE001  # the shared read below applies the caches' own Redis fallback
            verbose_router_logger.debug("routing prefetch failed, reading again: %s", e)
            return None
        results: Final[list[list[object | None] | None]] = []  # mutable-ok: filled per read below
        for cache, keys in reads:
            pending = await cache._prepare_batch_get(keys, local_only=True)  # pyright: ignore[reportPrivateUsage]  # same two-step read as async_batch_get_cache_shared
            if any(
                key not in prefetch.fetched for key, local_value in zip(keys, pending.result) if local_value is None
            ):
                return None
            missed = {key: values.get(key) for key, local in zip(keys, pending.result) if local is None}
            results.append(await cache._apply_batch_get(pending, missed))  # pyright: ignore[reportPrivateUsage]  # same two-step read as async_batch_get_cache_shared
        return results
