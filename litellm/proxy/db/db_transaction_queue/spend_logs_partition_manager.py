"""
Manages native Postgres range partitions for the LiteLLM_SpendLogs table.

At high request volume, retention via batched DELETE leaves dead tuples that
autovacuum cannot reclaim fast enough, so the table keeps growing on disk. When
the table is range-partitioned on startTime, dropping old data becomes a
DROP TABLE on a whole partition: an instant metadata operation that returns disk
to the OS immediately, with no tombstones and no vacuum.

This manager only acts when use_spend_logs_partitioning is enabled in
general_settings AND the table is already partitioned (set up via the
db_scripts/partition_spend_logs.sql runbook). Without both, the cleanup job
keeps the batched-DELETE path, so existing deployments are untouched.
"""

import re
from datetime import date, datetime, timedelta, timezone
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    SPEND_LOG_PARTITION_INTERVAL,
    SPEND_LOG_PARTITION_PRECREATE_AHEAD,
)

SPEND_LOGS_TABLE: Final = "LiteLLM_SpendLogs"

PartitionInterval = str  # "day" | "week" | "month"

VALID_PARTITION_INTERVALS: Final = {"day", "week", "month"}

_BOUND_UPPER_RE: Final = re.compile(r"TO \('([^']+)'\)")


def period_start(day: date, interval: PartitionInterval) -> date:
    """First day of the partition period that `day` falls into (UTC)."""
    if interval == "day":
        return day
    if interval == "week":
        return day - timedelta(days=day.weekday())
    if interval == "month":
        return day.replace(day=1)
    raise ValueError(f"Unsupported partition interval: {interval}")


def next_period_start(start: date, interval: PartitionInterval) -> date:
    if interval == "day":
        return start + timedelta(days=1)
    if interval == "week":
        return start + timedelta(days=7)
    if interval == "month":
        if start.month == 12:
            return start.replace(year=start.year + 1, month=1)
        return start.replace(month=start.month + 1)
    raise ValueError(f"Unsupported partition interval: {interval}")


def partition_name(start: date) -> str:
    return f"{SPEND_LOGS_TABLE}_p{start.strftime('%Y%m%d')}"


def upcoming_partitions(today: date, interval: PartitionInterval, ahead: int) -> list[tuple[str, date, date]]:
    """
    Specs (name, lower_inclusive, upper_exclusive) for the current period plus
    the next `ahead` periods, so writes always have a partition to land in.
    """
    specs: Final[list[tuple[str, date, date]]] = []
    start = period_start(today, interval)
    for _ in range(ahead + 1):
        upper = next_period_start(start, interval)
        specs.append((partition_name(start), start, upper))
        start = upper
    return specs


def parse_partition_upper_bound(bound_expr: str) -> datetime | None:
    """
    Upper bound of a Postgres partition from its `pg_get_expr(relpartbound)`
    string, e.g. "FOR VALUES FROM ('2026-06-01 00:00:00') TO ('2026-06-02 00:00:00')".
    Returns None for the DEFAULT partition or anything we cannot parse, so such
    partitions are never selected for dropping.
    """
    if "DEFAULT" in bound_expr.upper():
        return None
    match: Final = _BOUND_UPPER_RE.search(bound_expr)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(match.group(1))
    except ValueError:
        return None


def select_partitions_to_drop(partitions: list[tuple[str, datetime | None]], cutoff: datetime) -> list[str]:
    """
    Names of partitions whose entire range is older than `cutoff` (upper bound
    <= cutoff). `cutoff` and the bounds are UTC-naive. Partitions without a
    parseable upper bound (e.g. DEFAULT) are kept.
    """
    return [name for name, upper in partitions if upper is not None and upper <= cutoff]


class SpendLogsPartitionManager:
    def __init__(
        self,
        interval: PartitionInterval = SPEND_LOG_PARTITION_INTERVAL,
        precreate_ahead: int = SPEND_LOG_PARTITION_PRECREATE_AHEAD,
    ):
        if interval not in VALID_PARTITION_INTERVALS:
            verbose_proxy_logger.warning(
                "Invalid SPEND_LOG_PARTITION_INTERVAL %r, falling back to 'day'. Supported values: %s",
                interval,
                sorted(VALID_PARTITION_INTERVALS),
            )
            interval = "day"
        self.interval = interval
        self.precreate_ahead = precreate_ahead

    async def is_partitioned(self, prisma_client) -> bool:
        try:
            rows: Final = await prisma_client.db.query_raw(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_partitioned_table pt
                    JOIN pg_class c ON c.oid = pt.partrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = $1
                      AND n.nspname = current_schema()
                ) AS partitioned
                """,
                SPEND_LOGS_TABLE,
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "Could not determine if %s is partitioned, assuming it is not: %s",
                SPEND_LOGS_TABLE,
                e,
            )
            return False
        return bool(rows and rows[0].get("partitioned"))

    async def ensure_partitions(self, prisma_client) -> list[str]:
        """
        Ensure the current and upcoming partitions exist, returning the names
        now present. CREATE TABLE IF NOT EXISTS is a no-op for partitions that
        already exist, so this list is "ensured present", not "newly created".
        """
        ensured: Final[list[str]] = []
        for name, lower, upper in upcoming_partitions(
            datetime.now(timezone.utc).date(), self.interval, self.precreate_ahead
        ):
            try:
                await prisma_client.db.execute_raw(
                    f'CREATE TABLE IF NOT EXISTS "{name}" '
                    f'PARTITION OF "{SPEND_LOGS_TABLE}" '
                    f"FOR VALUES FROM ('{lower.isoformat()}') TO ('{upper.isoformat()}')"
                )
                ensured.append(name)
            except Exception as e:
                verbose_proxy_logger.warning("Failed to ensure spend-log partition %s: %s", name, e)
        return ensured

    async def _list_partitions(self, prisma_client) -> list[tuple[str, datetime | None]]:
        rows: Final = await prisma_client.db.query_raw(
            """
            SELECT c.relname AS name,
                   pg_get_expr(c.relpartbound, c.oid) AS bound
            FROM pg_inherits i
            JOIN pg_class c ON c.oid = i.inhrelid
            JOIN pg_class p ON p.oid = i.inhparent
            JOIN pg_namespace n ON n.oid = p.relnamespace
            WHERE p.relname = $1
              AND n.nspname = current_schema()
            """,
            SPEND_LOGS_TABLE,
        )
        return [(row["name"], parse_partition_upper_bound(row.get("bound") or "")) for row in rows]

    async def drop_partitions_older_than(self, prisma_client, cutoff: datetime) -> list[str]:
        """DROP every partition whose whole range is older than `cutoff`."""
        cutoff_naive: Final = cutoff.astimezone(timezone.utc).replace(tzinfo=None)
        partitions: Final = await self._list_partitions(prisma_client)
        to_drop: Final = select_partitions_to_drop(partitions, cutoff_naive)
        dropped: Final[list[str]] = []
        for name in to_drop:
            try:
                await prisma_client.db.execute_raw(f'DROP TABLE IF EXISTS "{name}"')
                dropped.append(name)
            except Exception as e:
                verbose_proxy_logger.warning("Failed to drop spend-log partition %s: %s", name, e)
        return dropped
