"""
In memory buffer for per-budget-window spend increments.

Kept separate from SpendUpdateQueue: an increment is only meaningful together
with the window it landed in, so two increments for the same entity must not be
merged when their window_start differs.
"""

import asyncio
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import chain, groupby
from typing import Final, TypedDict

from typing_extensions import ReadOnly

from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_ASYNCIO_QUEUE_MAXSIZE
from litellm.proxy.db.db_transaction_queue.base_update_queue import BaseUpdateQueue


class WindowSpendTransaction(TypedDict):
    """One increment for a single (entity, budget window) pair.

    window_start is an ISO-8601 string rather than a datetime so the
    transaction survives the JSON round trip through the Redis buffer.

    request_ids carries the LiteLLM_SpendLogs ids this spend came from. The
    one-time seed for a window that has no row yet subtracts them from its
    LiteLLM_SpendLogs aggregate, because the spend log writer flushes on its
    own ~2s poll and will usually have persisted these rows before the window
    queue flushes; without the exclusion the seed and the increment would each
    count them.

    started_at is the earliest request start in the batch. The seed only
    subtracts a request_id whose LiteLLM_SpendLogs.startTime is at or after it,
    so a client that replays an old id through x-litellm-call-id cannot make the
    seed drop the historical row that id already paid for.
    """

    entity_type: ReadOnly[str]
    entity_id: ReadOnly[str]
    window_duration: ReadOnly[str]
    window_start: ReadOnly[str]
    spend: ReadOnly[float]
    request_ids: ReadOnly[Sequence[str]]
    started_at: ReadOnly[str | None]


def to_naive_utc(value: datetime) -> datetime:
    """LiteLLM_BudgetWindowSpend.window_start is TIMESTAMP(3), which holds naive UTC."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def window_spend_group_key(transaction: WindowSpendTransaction) -> tuple[str, str, str, str]:
    """Identity of a window increment: the row's primary key plus the window it
    belongs to. Two increments only aggregate when all four match."""
    return (
        transaction["entity_type"],
        transaction["entity_id"],
        transaction["window_duration"],
        transaction["window_start"],
    )


def build_window_spend_transaction(
    entity_type: str,
    entity_id: str,
    window_duration: str,
    window_start: datetime,
    spend: float,
    request_id: str | None = None,
    started_at: datetime | None = None,
) -> WindowSpendTransaction:
    return WindowSpendTransaction(
        entity_type=entity_type,
        entity_id=entity_id,
        window_duration=window_duration,
        window_start=to_naive_utc(window_start).isoformat(timespec="microseconds"),
        spend=spend,
        request_ids=() if request_id is None else (request_id,),
        started_at=None
        if started_at is None
        else to_naive_utc(started_at.astimezone(timezone.utc)).isoformat(timespec="microseconds"),
    )


def _merge_window_spend_transactions(
    payloads: tuple[WindowSpendTransaction, ...],
) -> WindowSpendTransaction:
    first: Final = payloads[0]
    started_ats: Final = tuple(
        started_at for payload in payloads if (started_at := payload.get("started_at")) is not None
    )
    return WindowSpendTransaction(
        entity_type=first["entity_type"],
        entity_id=first["entity_id"],
        window_duration=first["window_duration"],
        window_start=first["window_start"],
        spend=math.fsum(payload["spend"] for payload in payloads),
        request_ids=tuple(sorted(frozenset(chain.from_iterable(payload["request_ids"] for payload in payloads)))),
        started_at=min(started_ats) if started_ats else None,
    )


class WindowSpendUpdateQueue(BaseUpdateQueue):
    """
    In memory buffer for budget-window spend increments committed to
    LiteLLM_BudgetWindowSpend.

    Add an update with the payload built by build_window_spend_transaction:
        window_spend_update_queue.add_update(
            build_window_spend_transaction(
                entity_type="key",
                entity_id="<hashed token>",
                window_duration="30d",
                window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
                spend=0.02,
            )
        )
    """

    def __init__(self) -> None:
        super().__init__()
        self.update_queue: asyncio.Queue[tuple[WindowSpendTransaction, ...]] = asyncio.Queue(
            maxsize=LITELLM_ASYNCIO_QUEUE_MAXSIZE
        )

    async def add_update(self, update: WindowSpendTransaction) -> None:
        """Enqueue an update."""
        verbose_proxy_logger.debug("Adding budget window spend update to queue: %s", update)
        await self.update_queue.put((update,))
        if self.update_queue.qsize() >= self.MAX_SIZE_IN_MEMORY_QUEUE:
            verbose_proxy_logger.warning(
                "Budget window spend update queue is full. Aggregating all entries in queue to concatenate entries."
            )
            await self.aggregate_queue_updates()

    async def aggregate_queue_updates(self) -> None:
        """Collapse everything currently queued into a single aggregated update."""
        updates: Final = await self.flush_all_updates_from_in_memory_queue()
        await self.update_queue.put(WindowSpendUpdateQueue.get_aggregated_window_spend_transactions(updates))

    async def flush_and_get_aggregated_window_spend_transactions(
        self,
    ) -> tuple[WindowSpendTransaction, ...]:
        """Drain the queue and return the increments aggregated per window."""
        updates: Final = await self.flush_all_updates_from_in_memory_queue()
        if len(updates) > 0:
            verbose_proxy_logger.info(
                "Spend tracking - flushed %d budget window spend update batches from in-memory queue",
                len(updates),
            )
        return WindowSpendUpdateQueue.get_aggregated_window_spend_transactions(updates)

    @staticmethod
    def get_aggregated_window_spend_transactions(
        updates: Sequence[Sequence[WindowSpendTransaction]],
    ) -> tuple[WindowSpendTransaction, ...]:
        """Sum spend per (entity_type, entity_id, window_duration, window_start).

        Increments belonging to different windows stay separate even when they
        share a primary key, so a window boundary crossed mid-tick does not fold
        the new window's spend into the previous window's total.

        The result is ordered by that same key, which is the order the flush
        needs: primary key first for cross-pod lock ordering, then window_start
        so an older window is applied before the roll that supersedes it.
        """
        ordered: Final = tuple(sorted(chain.from_iterable(updates), key=window_spend_group_key))
        return tuple(
            _merge_window_spend_transactions(tuple(group)) for _, group in groupby(ordered, key=window_spend_group_key)
        )
