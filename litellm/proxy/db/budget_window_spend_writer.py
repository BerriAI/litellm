"""
Writer for LiteLLM_BudgetWindowSpend.

The table holds one row per configured budget window whose window_start rolls
forward in place, so budget enforcement can read a maintained running total
instead of aggregating LiteLLM_SpendLogs every time a window counter goes cold
(issue #35766). Raw SQL rather than the Prisma upsert helper because the
conditional roll cannot be expressed through the query builder.

Seeding a row that does not exist yet reads LiteLLM_SpendLogs once, excluding
the requests whose increments are in the same batch so neither source counts
them twice. One gap survives that exclusion: without the Redis transaction
buffer every pod flushes its own increments, so a row seeded by one pod can
miss increments still queued on another. That is bounded by a single flush
interval, happens at most once per window row, and errs toward under-counting
for that interval only.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
    WindowSpendTransaction,
    to_naive_utc,
    window_spend_group_key,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


_SELECT_EXISTING_ROWS_SQL: Final = (
    'SELECT entity_type, entity_id, window_duration FROM "LiteLLM_BudgetWindowSpend" '
    "WHERE (entity_type, entity_id, window_duration) "
    "IN (SELECT * FROM unnest($1::text[], $2::text[], $3::text[]))"
)

_UPSERT_WINDOW_SPEND_SQL: Final = (
    'INSERT INTO "LiteLLM_BudgetWindowSpend" '
    "(entity_type, entity_id, window_duration, window_start, spend, created_at, updated_at) "
    "VALUES ($1, $2, $3, ($4::timestamptz AT TIME ZONE 'UTC'), $5, "
    "($7::timestamptz AT TIME ZONE 'UTC'), ($7::timestamptz AT TIME ZONE 'UTC')) "
    "ON CONFLICT (entity_type, entity_id, window_duration) DO UPDATE SET "
    "spend = CASE "
    'WHEN "LiteLLM_BudgetWindowSpend".window_start >= EXCLUDED.window_start '
    'THEN "LiteLLM_BudgetWindowSpend".spend + $6 '
    "ELSE EXCLUDED.spend "
    "END, "
    'window_start = GREATEST("LiteLLM_BudgetWindowSpend".window_start, EXCLUDED.window_start), '
    "updated_at = ($7::timestamptz AT TIME ZONE 'UTC')"
)

_ROLL_WINDOW_SPEND_SQL: Final = (
    'UPDATE "LiteLLM_BudgetWindowSpend" SET '
    "window_start = ($4::timestamptz AT TIME ZONE 'UTC'), "
    "spend = 0, "
    "updated_at = ($5::timestamptz AT TIME ZONE 'UTC') "
    "WHERE entity_type = $1 AND entity_id = $2 AND window_duration = $3 "
    "AND window_start < ($4::timestamptz AT TIME ZONE 'UTC')"
)

_SEED_FROM_SPEND_LOGS_KEY_SQL: Final = (
    'SELECT COALESCE(SUM(spend), 0.0) AS total FROM "LiteLLM_SpendLogs" '
    "WHERE api_key = $1 AND \"startTime\" >= ($2::timestamptz AT TIME ZONE 'UTC') "
    "AND NOT (request_id = ANY($3::text[]) AND \"startTime\" >= ($4::timestamptz AT TIME ZONE 'UTC'))"
)

_SEED_FROM_SPEND_LOGS_TEAM_SQL: Final = (
    'SELECT COALESCE(SUM(spend), 0.0) AS total FROM "LiteLLM_SpendLogs" '
    "WHERE team_id = $1 AND \"startTime\" >= ($2::timestamptz AT TIME ZONE 'UTC') "
    "AND NOT (request_id = ANY($3::text[]) AND \"startTime\" >= ($4::timestamptz AT TIME ZONE 'UTC'))"
)

_SEED_FROM_SPEND_LOGS_KEY_UNBOUNDED_SQL: Final = (
    'SELECT COALESCE(SUM(spend), 0.0) AS total FROM "LiteLLM_SpendLogs" '
    "WHERE api_key = $1 AND \"startTime\" >= ($2::timestamptz AT TIME ZONE 'UTC')"
)

_SEED_FROM_SPEND_LOGS_TEAM_UNBOUNDED_SQL: Final = (
    'SELECT COALESCE(SUM(spend), 0.0) AS total FROM "LiteLLM_SpendLogs" '
    "WHERE team_id = $1 AND \"startTime\" >= ($2::timestamptz AT TIME ZONE 'UTC')"
)

_UPSERT_TRANSACTION_TIMEOUT: Final = timedelta(seconds=60)


class WindowSpendLogsAggregate(Protocol):
    """Sums LiteLLM_SpendLogs for one entity since window_start, ignoring the
    requests whose ids are handed in.

    Injected so the flush can be exercised without a database and so the
    expensive aggregate stays swappable.
    """

    async def __call__(
        self,
        prisma_client: "PrismaClient",
        entity_type: str,
        entity_id: str,
        window_start: datetime,
        exclude_request_ids: Sequence[str],
        exclude_started_at: datetime | None,
    ) -> float | None: ...


async def spend_logs_total_excluding(
    prisma_client: "PrismaClient",
    entity_type: str,
    entity_id: str,
    window_start: datetime,
    exclude_request_ids: Sequence[str],
    exclude_started_at: datetime | None,
) -> float | None:
    """LiteLLM_SpendLogs spend for one entity since window_start, minus the
    requests already accounted for by the increments being flushed.

    The spend log writer drains its own queue on a ~2s poll whenever anything
    is queued, while window increments flush on the much slower batch tick, so
    by the time a window row is seeded its batch's log rows are normally
    already in the table. Counting them in the seed and again in the increment
    is what made a fresh row land at twice the true spend.

    The exclusion is bounded to rows that started at or after the batch's
    earliest request. request_id can be chosen by the client
    (x-litellm-call-id), so an unbounded exclusion would let a replayed old id
    erase a historical row from the seed while its increment still lands.
    Without a known start the batch's ids are not excluded at all: that can
    only over-count once, which enforcement tolerates, whereas under-counting
    is a budget bypass.
    """
    if entity_type == Litellm_EntityType.KEY.value:
        bounded_sql, unbounded_sql = _SEED_FROM_SPEND_LOGS_KEY_SQL, _SEED_FROM_SPEND_LOGS_KEY_UNBOUNDED_SQL
    elif entity_type == Litellm_EntityType.TEAM.value:
        bounded_sql, unbounded_sql = _SEED_FROM_SPEND_LOGS_TEAM_SQL, _SEED_FROM_SPEND_LOGS_TEAM_UNBOUNDED_SQL
    else:
        return None
    rows: Final = (
        await prisma_client.db.query_raw(unbounded_sql, entity_id, window_start)
        if exclude_started_at is None or not exclude_request_ids
        else await prisma_client.db.query_raw(
            bounded_sql,
            entity_id,
            window_start,
            tuple(exclude_request_ids),
            _exclusion_lower_bound(exclude_started_at),
        )
    )
    if not rows:
        return 0.0
    return float(rows[0].get("total") or 0.0)


def _exclusion_lower_bound(started_at: datetime) -> datetime:
    """LiteLLM_SpendLogs.startTime is TIMESTAMP(3); floor to the second so a
    millisecond rounding of the batch's own earliest row cannot slip under it."""
    return to_naive_utc(started_at).replace(microsecond=0)


def _primary_key(transaction: WindowSpendTransaction) -> tuple[str, str, str]:
    return (
        transaction["entity_type"],
        transaction["entity_id"],
        transaction["window_duration"],
    )


async def _existing_primary_keys(
    prisma_client: "PrismaClient",
    transactions: tuple[WindowSpendTransaction, ...],
) -> frozenset[tuple[str, str, str]]:
    rows: Final = await prisma_client.db.query_raw(
        _SELECT_EXISTING_ROWS_SQL,
        tuple(transaction["entity_type"] for transaction in transactions),
        tuple(transaction["entity_id"] for transaction in transactions),
        tuple(transaction["window_duration"] for transaction in transactions),
    )
    return frozenset((row["entity_type"], row["entity_id"], row["window_duration"]) for row in rows or ())


async def _seed_base_for_missing_row(
    prisma_client: "PrismaClient",
    transaction: WindowSpendTransaction,
    existing_primary_keys: frozenset[tuple[str, str, str]],
    spend_logs_aggregate: WindowSpendLogsAggregate,
) -> float:
    """Spend already recorded for a window that has no row yet.

    This is the LiteLLM_SpendLogs aggregate the window counter reseed runs on
    every cold counter today, but here it runs once per window lifetime and off
    the request path, and it excludes this batch's own requests so they are
    counted by their increments alone.
    """
    if _primary_key(transaction) in existing_primary_keys:
        return 0.0
    base: Final = await spend_logs_aggregate(
        prisma_client=prisma_client,
        entity_type=transaction["entity_type"],
        entity_id=transaction["entity_id"],
        window_start=datetime.fromisoformat(transaction["window_start"]).replace(tzinfo=timezone.utc),
        exclude_request_ids=transaction["request_ids"],
        exclude_started_at=_transaction_started_at(transaction),
    )
    return float(base or 0.0)


def _transaction_started_at(transaction: WindowSpendTransaction) -> datetime | None:
    started_at: Final = transaction.get("started_at")
    if started_at is None:
        return None
    return datetime.fromisoformat(started_at).replace(tzinfo=timezone.utc)


def _upsert_params(
    transaction: WindowSpendTransaction,
    seed_base: float,
    now: datetime,
) -> tuple[str, str, str, datetime, float, float, datetime]:
    """$5 is what a brand new row starts at (pre-existing spend plus this
    increment); $6 is the increment alone, which is all an already-current row
    may add. They are equal for every row that already existed, so a row is
    never seeded twice when two pods flush the same new window."""
    increment: Final = float(transaction["spend"])
    return (
        transaction["entity_type"],
        transaction["entity_id"],
        transaction["window_duration"],
        datetime.fromisoformat(transaction["window_start"]),
        seed_base + increment,
        increment,
        now,
    )


async def commit_window_spend_updates(
    prisma_client: "PrismaClient",
    transactions: Sequence[WindowSpendTransaction],
    spend_logs_aggregate: WindowSpendLogsAggregate = spend_logs_total_excluding,
) -> None:
    """Apply aggregated window increments to LiteLLM_BudgetWindowSpend.

    An increment at or behind the row's window_start adds into the row (this is
    how in-flight requests that raced a reset carry into the new window); an
    increment ahead of it rolls the window and starts from that increment.

    Statements are ordered by primary key so concurrent pods take row locks in
    the same order, with window_start breaking ties so an older window is
    applied before the roll that supersedes it.
    """
    if not transactions:
        return

    ordered: Final = tuple(sorted(transactions, key=window_spend_group_key))
    existing_primary_keys: Final = await _existing_primary_keys(
        prisma_client=prisma_client,
        transactions=ordered,
    )
    seed_bases: Final = tuple(
        [
            await _seed_base_for_missing_row(
                prisma_client=prisma_client,
                transaction=transaction,
                existing_primary_keys=existing_primary_keys,
                spend_logs_aggregate=spend_logs_aggregate,
            )
            for transaction in ordered
        ]
    )

    now: Final = to_naive_utc(datetime.now(timezone.utc))
    verbose_proxy_logger.debug(
        "Spend tracking - committing %d budget window spend upserts over %d existing rows",
        len(ordered),
        len(existing_primary_keys),
    )
    async with prisma_client.db.tx(timeout=_UPSERT_TRANSACTION_TIMEOUT) as db_transaction:
        async with db_transaction.batch_() as batcher:
            for transaction, seed_base in zip(ordered, seed_bases):
                batcher.execute_raw(
                    _UPSERT_WINDOW_SPEND_SQL,
                    *_upsert_params(transaction=transaction, seed_base=seed_base, now=now),
                )


async def roll_window_spend_row(
    prisma_client: "PrismaClient",
    entity_type: str,
    entity_id: str,
    window_duration: str,
    new_window_start: datetime,
) -> None:
    """Move a row onto the window that just started and zero its spend.

    Conditional on the stored window_start still being behind the new one so a
    pod that already rolled the row (or increments that arrived under the new
    window) are not clobbered.
    """
    await prisma_client.db.execute_raw(
        _ROLL_WINDOW_SPEND_SQL,
        entity_type,
        entity_id,
        window_duration,
        to_naive_utc(new_window_start),
        to_naive_utc(datetime.now(timezone.utc)),
    )
