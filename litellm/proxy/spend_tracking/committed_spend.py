"""Committed spend counters: twins of the entity spend counters that only ever take charged cost, never budget holds.

The remaining budget gauges read them, so a request's worst-case reservation never shows as spent. They are kept
only for entities with a budget, while a ``PrometheusLogger`` is registered.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.spend_tracking.spend_counter_batch import (
    SPEND_COUNTERS_TARGET,
    admission_counter_keys,
    bind_spend_counter_keys,
    read_batched_spend_counter,
)
from litellm.types.proxy.carried_budget_state import (
    END_USER_COUNTER_PREFIX,
    CommittedBudgetSpend,
    EntitySpendCounterKeys,
)

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.utils import PrismaClient

_BASE_COUNTER_PREFIXES: Final = ("spend:key:", "spend:team:", "spend:user:", "spend:org:", END_USER_COUNTER_PREFIX)
_FETCHED_VALUES: Final = TypeAdapter(dict[str, float | None])
_STORED_VALUE: Final[TypeAdapter[float | None]] = TypeAdapter(float | None)


def committed_spend_key(counter_key: str) -> str:
    return f"committed_{counter_key}"


def _has_committed_twin(counter_key: str) -> bool:
    return counter_key.startswith(_BASE_COUNTER_PREFIXES) and ":window:" not in counter_key


def committed_spend_keys(counter_keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(committed_spend_key(counter_key) for counter_key in counter_keys if _has_committed_twin(counter_key))


def committed_spend_tracked() -> bool:
    return litellm.logging_callback_manager.callback_is_active(callback_type=PrometheusLogger)


def bind_committed_admission_counters(token: UserAPIKeyAuth, end_user_id: str | None) -> None:
    if committed_spend_tracked():
        bind_spend_counter_keys(frozenset(committed_spend_keys(admission_counter_keys(token, end_user_id))))


def _in_memory_committed_value(spend_counter_cache: DualCache, twin_key: str) -> float | None:
    return _STORED_VALUE.validate_python(
        spend_counter_cache.in_memory_cache.get_cache(key=twin_key)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped cache API
    )


async def _read_committed_values(spend_counter_cache: DualCache, twin_keys: tuple[str, ...]) -> Mapping[str, float]:
    redis_cache: Final = spend_counter_cache.redis_cache
    if redis_cache is None:
        stored: Final = {twin_key: _in_memory_committed_value(spend_counter_cache, twin_key) for twin_key in twin_keys}
        return {twin_key: value for twin_key, value in stored.items() if value is not None}
    fetched: Final = _FETCHED_VALUES.validate_python(
        await redis_cache.async_batch_get_cache(key_list=list(twin_keys))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped cache API
    )
    return {twin_key: value for twin_key, value in fetched.items() if value is not None}


def _committed_budget_spend(
    counter_keys: EntitySpendCounterKeys, committed: Mapping[str, float]
) -> CommittedBudgetSpend:
    def spend_of(counter_key: str | None) -> float | None:
        return committed.get(committed_spend_key(counter_key)) if counter_key is not None else None

    return CommittedBudgetSpend(
        counter_keys=counter_keys,
        key=spend_of(counter_keys.key),
        team=spend_of(counter_keys.team),
        user=spend_of(counter_keys.user),
        org=spend_of(counter_keys.org),
        end_user=spend_of(counter_keys.end_user),
    )


@with_service_target(SPEND_COUNTERS_TARGET)
async def read_committed_budget_spend(
    spend_counter_cache: DualCache, counter_keys: EntitySpendCounterKeys
) -> CommittedBudgetSpend:
    twin_keys: Final = committed_spend_keys(counter_keys.tracked())
    if not twin_keys:
        return CommittedBudgetSpend(counter_keys=counter_keys)
    try:
        committed: Final = await _read_committed_values(spend_counter_cache=spend_counter_cache, twin_keys=twin_keys)
    except Exception as e:  # noqa: BLE001  # a gauge read never fails the request; the gauges keep the stored spend
        verbose_proxy_logger.debug("committed spend read failed for %s: %s", twin_keys, e)
        return CommittedBudgetSpend(counter_keys=counter_keys)
    return _committed_budget_spend(counter_keys=counter_keys, committed=committed)


async def read_admission_committed_spend(
    spend_counter_cache: DualCache, counter_keys: EntitySpendCounterKeys
) -> CommittedBudgetSpend:
    """Rides the admission MGET: a counter the batch cannot answer stays unknown here and the emit-time read has it."""
    if spend_counter_cache.redis_cache is None:
        return await read_committed_budget_spend(spend_counter_cache, counter_keys)
    twin_keys: Final = committed_spend_keys(counter_keys.tracked())
    batched: Final = await asyncio.gather(*(read_batched_spend_counter(twin_key) for twin_key in twin_keys))
    return _committed_budget_spend(
        counter_keys=counter_keys,
        committed={
            twin_key: read[0] for twin_key, read in zip(twin_keys, batched) if read is not None and read[0] is not None
        },
    )


async def drop_committed_spend(spend_counter_cache: DualCache, counter_keys: Sequence[str]) -> None:
    """After a direct write to the stored spend the twin holds the old total, so the next charge seeds it again."""
    try:
        await spend_counter_cache.async_delete_cache_keys(committed_spend_keys(counter_keys))
    except Exception as e:  # noqa: BLE001  # the write already committed; a kept twin only lasts until its TTL
        verbose_proxy_logger.warning("committed spend drop failed: %s", e)


@with_service_target(SPEND_COUNTERS_TARGET)
async def seed_committed_spend(
    spend_counter_cache: DualCache, prisma_client: PrismaClient | None, counter_key: str
) -> bool:
    db_spend: Final = await (
        SpendCounterReseed.end_user_from_db(prisma_client=prisma_client, counter_key=counter_key)
        if counter_key.startswith(END_USER_COUNTER_PREFIX)
        else SpendCounterReseed.from_db(prisma_client=prisma_client, counter_key=counter_key)
    )
    if db_spend is None:
        return False
    twin_key: Final = committed_spend_key(counter_key)
    redis_cache: Final = spend_counter_cache.redis_cache
    if redis_cache is None:
        if _in_memory_committed_value(spend_counter_cache, twin_key) is None:
            spend_counter_cache.in_memory_cache.set_cache(key=twin_key, value=db_spend)  # pyright: ignore[reportUnknownMemberType]  # untyped cache API
        return True
    try:
        await redis_cache.async_set_max(key=twin_key, value=db_spend)
    except Exception as e:  # noqa: BLE001  # an unseeded twin only skips this charge; the next one seeds it again
        verbose_proxy_logger.debug("committed spend seed failed for %s: %s", twin_key, e)
        return False
    return True
