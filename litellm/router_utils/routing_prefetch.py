from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from opentelemetry.trace import Span

from litellm._internal_context import service_target
from litellm._logging import verbose_router_logger
from litellm.caching.redis_batch import Batch, active_request_redis_batches
from litellm.redis.stores.router_cooldowns import ROUTER_COOLDOWNS_TARGET, ActiveCooldowns, RouterCooldowns
from litellm.redis.stores.router_usage import ROUTER_USAGE_TARGET, RouterUsage, UsageCounter, UsageRead

_USAGE_SLOT: Final = "routing_read"
_COOLDOWNS_SLOT: Final = "routing_cooldowns"


@dataclass(frozen=True, slots=True)
class ArmedUsage:
    counters: tuple[UsageCounter, ...]
    read: UsageRead


def arm(store: RouterUsage, counters: Sequence[UsageCounter]) -> None:
    """The usage read a usage-based-routing-v2 pick takes later, declared on the request's native pre-call batch."""
    request: Final = active_request_redis_batches()
    if request is None or not store.has_redis or _USAGE_SLOT in request.prefetched:
        return
    request.prefetched[_USAGE_SLOT] = ArmedUsage(tuple(counters), store.usage(counters, batch=Batch.PRE_CALL))


@dataclass(frozen=True, slots=True)
class ArmedCooldowns:
    model_ids: frozenset[str]
    result: Awaitable[ActiveCooldowns]


def arm_cooldowns(store: RouterCooldowns, model_ids: Sequence[str]) -> None:
    request: Final = active_request_redis_batches()
    if request is None or not store.has_redis or _COOLDOWNS_SLOT in request.prefetched:
        return
    result: Final = store.active(model_ids, batch=Batch.PRE_CALL)
    request.prefetched[_COOLDOWNS_SLOT] = ArmedCooldowns(frozenset(model_ids), result)


def armed() -> bool:
    request: Final = active_request_redis_batches()
    return request is not None and (_USAGE_SLOT in request.prefetched or _COOLDOWNS_SLOT in request.prefetched)


async def _armed_cooldowns(model_ids: Sequence[str]) -> ActiveCooldowns | None:
    request: Final = active_request_redis_batches()
    taken: Final = None if request is None else request.prefetched.pop(_COOLDOWNS_SLOT, None)
    if not isinstance(taken, ArmedCooldowns):
        return None
    try:
        active: Final = await taken.result
    except Exception as e:  # noqa: BLE001  # the read falls back to the store's own Redis read
        verbose_router_logger.debug("cooldown prefetch failed, reading again: %s", e)
        return None
    if not taken.model_ids.issuperset(model_ids):
        return None
    wanted: Final = frozenset(model_ids)
    return tuple((model_id, cooldown) for model_id, cooldown in active if model_id in wanted)


async def read_cooldowns(
    store: RouterCooldowns, model_ids: Sequence[str], parent_otel_span: Span | None
) -> ActiveCooldowns:
    """The armed cooldown read when it covers ``model_ids``, taken once; otherwise the store reads on its own."""
    with service_target(ROUTER_COOLDOWNS_TARGET):
        prefetched: Final = await _armed_cooldowns(model_ids)
    if prefetched is not None:
        return prefetched
    return await store.active(model_ids, batch=Batch.NONE)


async def _armed_usage(counters: Sequence[UsageCounter]) -> Sequence[float | None] | None:
    request: Final = active_request_redis_batches()
    taken: Final = None if request is None else request.prefetched.pop(_USAGE_SLOT, None)
    if not isinstance(taken, ArmedUsage):
        return None
    if not frozenset(taken.counters).issuperset(counters):
        taken.read.release()
        return None
    try:
        values: Final = await taken.read.result
    except Exception as e:  # noqa: BLE001  # the read falls back to the store's own Redis read
        verbose_router_logger.debug("routing prefetch failed, reading again: %s", e)
        return None
    read: Final = dict(zip(taken.counters, values))
    return [read[counter] for counter in counters]


async def read(
    store: RouterUsage, counters: Sequence[UsageCounter], parent_otel_span: Span | None
) -> Sequence[float | None]:
    """The armed usage read when it covers ``counters``, taken once; otherwise the store reads on its own."""
    with service_target(ROUTER_USAGE_TARGET):
        prefetched: Final = await _armed_usage(counters)
    if prefetched is not None:
        return prefetched
    return await store.usage(counters, batch=Batch.NONE).result
