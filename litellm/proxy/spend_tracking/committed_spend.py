"""Committed spend counters: twins of the entity spend counters that only ever take charged cost, never budget holds.

The remaining budget gauges read them, so a request's worst-case reservation never shows as spent. They are kept
only for entities with a budget, while a ``PrometheusLogger`` is registered, in the same ``SpendCounterStore`` as the
counters they twin.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_batch import Batch
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.spend_tracking.spend_counter_batch import (
    admission_counter_keys,
    bind_spend_counter_keys,
    read_batched_spend_counter,
)
from litellm.redis.stores.spend_counters import (
    COMMITTED_COUNTER_PREFIX,
    END_USER_COUNTER_PREFIX,
    SPEND_COUNTERS_TARGET,
    SpendCounterStore,
    committed_counter,
)
from litellm.types.proxy.carried_budget_state import CommittedBudgetSpend, EntitySpendCounterKeys

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_BASE_COUNTER_PREFIXES: Final = ("spend:key:", "spend:team:", "spend:user:", "spend:org:", END_USER_COUNTER_PREFIX)


def committed_spend_key(counter_key: str) -> str:
    return committed_counter(counter_key)


def has_committed_twin(counter_key: str) -> bool:
    return counter_key.startswith(_BASE_COUNTER_PREFIXES) and ":window:" not in counter_key


def committed_spend_keys(counter_keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(committed_spend_key(counter_key) for counter_key in counter_keys if has_committed_twin(counter_key))


def is_committed_spend_key(counter_key: str) -> bool:
    return counter_key.startswith(COMMITTED_COUNTER_PREFIX)


def committed_spend_tracked() -> bool:
    return litellm.logging_callback_manager.callback_is_active(callback_type=PrometheusLogger)


def bind_committed_admission_counters(token: UserAPIKeyAuth, end_user_id: str | None) -> None:
    if committed_spend_tracked():
        bind_spend_counter_keys(frozenset(committed_spend_keys(admission_counter_keys(token, end_user_id))))


async def _read_committed_values(
    spend_counter_store: SpendCounterStore, twin_keys: tuple[str, ...]
) -> Mapping[str, float]:
    batched: Final = await spend_counter_store.read_many(twin_keys, batch=Batch.NONE)
    if batched is not None:
        return {twin_key: value for twin_key, value in batched.items() if value is not None}
    reads: Final = await asyncio.gather(*(spend_counter_store.read(twin_key) for twin_key in twin_keys))
    return {twin_key: value for twin_key, (value, _) in zip(twin_keys, reads, strict=True) if value is not None}


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
    spend_counter_store: SpendCounterStore, counter_keys: EntitySpendCounterKeys
) -> CommittedBudgetSpend:
    twin_keys: Final = committed_spend_keys(counter_keys.tracked())
    if not twin_keys:
        return CommittedBudgetSpend(counter_keys=counter_keys)
    try:
        committed: Final = await _read_committed_values(spend_counter_store, twin_keys)
    except Exception as e:  # noqa: BLE001  # a gauge read never fails the request; the gauges keep the stored spend
        verbose_proxy_logger.debug("committed spend read failed for %s: %s", twin_keys, e)
        return CommittedBudgetSpend(counter_keys=counter_keys)
    return _committed_budget_spend(counter_keys=counter_keys, committed=committed)


async def warm_committed_counter_keys(
    spend_counter_store: SpendCounterStore, counter_keys: EntitySpendCounterKeys
) -> EntitySpendCounterKeys:
    """For a charge with no auth to name the budgeted entities: only a budgeted entity's twin exists."""
    committed: Final = await read_committed_budget_spend(spend_counter_store, counter_keys)
    return EntitySpendCounterKeys(
        key=counter_keys.key if committed.key is not None else None,
        team=counter_keys.team if committed.team is not None else None,
        user=counter_keys.user if committed.user is not None else None,
        org=counter_keys.org if committed.org is not None else None,
        end_user=counter_keys.end_user if committed.end_user is not None else None,
    )


async def read_admission_committed_spend(
    spend_counter_store: SpendCounterStore, counter_keys: EntitySpendCounterKeys
) -> CommittedBudgetSpend:
    """Rides the admission MGET: a counter the batch cannot answer stays unknown here and the emit-time read has it."""
    if not spend_counter_store.has_redis:
        return await read_committed_budget_spend(spend_counter_store, counter_keys)
    twin_keys: Final = committed_spend_keys(counter_keys.tracked())
    batched: Final = await asyncio.gather(*(read_batched_spend_counter(twin_key) for twin_key in twin_keys))
    return _committed_budget_spend(
        counter_keys=counter_keys,
        committed={
            twin_key: read[0] for twin_key, read in zip(twin_keys, batched) if read is not None and read[0] is not None
        },
    )


async def drop_committed_spend(spend_counter_store: SpendCounterStore, counter_keys: Sequence[str]) -> None:
    """After a direct write to the stored spend the twin holds the old total, so the next charge seeds it again."""
    twin_keys: Final = committed_spend_keys(counter_keys)
    if not twin_keys:
        return
    try:
        await spend_counter_store.delete(twin_keys)
    except Exception as e:  # noqa: BLE001  # the write already committed; a kept twin lasts until charges stop for its TTL
        verbose_proxy_logger.warning("committed spend drop failed: %s", e)


@with_service_target(SPEND_COUNTERS_TARGET)
async def seed_committed_spend(
    spend_counter_store: SpendCounterStore, prisma_client: PrismaClient | None, counter_key: str
) -> bool:
    """One DB read per worker for a cold twin: charges waiting on the lock find the twin the first one wrote."""
    twin_key: Final = committed_spend_key(counter_key)
    async with await SpendCounterReseed.counter_lock(twin_key):
        try:
            stored, _ = await spend_counter_store.read(twin_key)
        except Exception as e:  # noqa: BLE001  # an unseeded twin only skips this charge; the next one seeds it again
            verbose_proxy_logger.debug("committed spend seed read failed for %s: %s", twin_key, e)
            return False
        return stored is not None or await _seed_committed_spend_from_db(
            spend_counter_store, prisma_client, counter_key
        )


async def _seed_committed_spend_from_db(
    spend_counter_store: SpendCounterStore, prisma_client: PrismaClient | None, counter_key: str
) -> bool:
    db_spend: Final = await (
        SpendCounterReseed.end_user_from_db(prisma_client=prisma_client, counter_key=counter_key)
        if counter_key.startswith(END_USER_COUNTER_PREFIX)
        else SpendCounterReseed.from_db(prisma_client=prisma_client, counter_key=counter_key)
    )
    if db_spend is None:
        return False
    twin_key: Final = committed_spend_key(counter_key)
    try:
        await spend_counter_store.seed(twin_key, db_spend, increment_if_lost=False)
    except Exception as e:  # noqa: BLE001  # an unseeded twin only skips this charge; the next one seeds it again
        verbose_proxy_logger.debug("committed spend seed failed for %s: %s", twin_key, e)
        return False
    return True
