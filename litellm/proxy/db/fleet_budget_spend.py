"""Replicated spend read for admission. Regional spend counters stay local."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from time import monotonic
from typing import TYPE_CHECKING, Final
from weakref import WeakKeyDictionary, WeakSet

from litellm.constants import PROXY_DB_LOOKUP_DEADLINE_SECONDS
from litellm.proxy.common_utils.timezone_utils import get_budget_window_reset_fields
from litellm.proxy.db.db_lookup_gate import bounded_db_lookup, db_lookup_gate
from litellm.repositories.budget_limits_repository import BudgetLimitsTransaction

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_SCOPE_COLUMNS: Final[Mapping[str, str]] = {
    "key": "api_key",
    "team": "team_id",
    "user": "user",
    "end_user": "end_user",
}
_WINDOW_SCOPES: Final[Mapping[str, str]] = {"Key": "key", "Team": "team", "User": "user"}
_MAX_ENTITY_ID_BYTES: Final = 2048
_ENTITY_DIGEST_PREFIX: Final = "sha256:"
_INACTIVE_RECHECK_SECONDS: Final = 5
_inactive_until: Final[WeakKeyDictionary[PrismaClient, float]] = WeakKeyDictionary()
_started: Final[WeakSet[PrismaClient]] = WeakSet()


def replicated_budget_entity_id(identifier: str | None) -> str | None:
    """Use the same bounded identifier in published records and budget lookups."""
    if identifier is None:
        return None
    encoded: Final = identifier.encode()
    if len(encoded) > _MAX_ENTITY_ID_BYTES or identifier.startswith(_ENTITY_DIGEST_PREFIX):
        return _ENTITY_DIGEST_PREFIX + sha256(encoded).hexdigest()
    return identifier


def primary_budget_window_start(duration: str | None, reset_at: datetime | None) -> datetime | None:
    if duration is None or reset_at is None:
        return None
    reset: Final = reset_at.replace(tzinfo=reset_at.tzinfo or timezone.utc)
    return datetime.fromisoformat(get_budget_window_reset_fields(duration, reset_at=reset)["window_start"])


async def fleet_budget_replication_started(prisma_client: PrismaClient | None) -> bool:
    from prisma.errors import RawQueryError

    from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
    from litellm.proxy.utils import PrismaClient, query_raw_rows

    if not isinstance(prisma_client, PrismaClient):
        return False
    if prisma_client in _started:
        return True
    now: Final = monotonic()
    if _inactive_until.get(prisma_client, 0.0) > now:
        return False
    try:
        rows: Final = await query_raw_rows(
            prisma_client.writer_db,
            'SELECT EXISTS (SELECT 1 FROM "LiteLLM_SpendLogReplication" WHERE id = 1) AS started',
        )
    except RawQueryError as error:
        if PrismaDBExceptionHandler.postgres_sqlstate(error) != "42P01":
            raise
    else:
        if rows and rows[0].get("started") is True:
            _started.add(prisma_client)
            return True
    _inactive_until[prisma_client] = now + _INACTIVE_RECHECK_SECONDS
    return False


async def record_fleet_key_budget_reset(tx: BudgetLimitsTransaction, api_key: str, now: datetime) -> None:
    await tx.execute_raw(
        """INSERT INTO "LiteLLM_FleetKeyBudgetReset" ("api_key", "reset_at")
            VALUES ($1, $2::text::timestamptz AT TIME ZONE 'UTC')
            ON CONFLICT ("api_key") DO UPDATE SET "reset_at" =
                GREATEST("LiteLLM_FleetKeyBudgetReset"."reset_at", EXCLUDED."reset_at")""",
        api_key,
        now.isoformat(),
    )


async def read_replica_budget_spend(
    prisma_client: PrismaClient | None,
    counter_key: str,
    *,
    window_entity_type: str | None = None,
    window_entity_id: str | None = None,
    window_start: datetime | None = None,
) -> float:
    from litellm.proxy.utils import PrismaClient

    scope_and_id: Final = counter_key.partition(":")[2].partition(":")
    use_window_scope: Final = window_entity_type is not None and window_entity_id is not None
    scope: Final = _WINDOW_SCOPES.get(window_entity_type or "", "") if use_window_scope else scope_and_id[0]
    entity_id: Final = window_entity_id if use_window_scope else scope_and_id[2]
    column: Final = _SCOPE_COLUMNS.get(scope)
    if not isinstance(prisma_client, PrismaClient) or column is None or not entity_id:
        return 0.0
    return await bounded_db_lookup(
        _read_replica_budget_spend(prisma_client, column, entity_id, window_start),
        name="fleet_budget_spend",
    )


async def _read_replica_budget_spend(
    prisma_client: PrismaClient,
    column: str,
    entity_id: str,
    window_start: datetime | None,
) -> float:
    cutoff: Final = (
        window_start.replace(tzinfo=window_start.tzinfo or timezone.utc).isoformat()
        if window_start is not None
        else None
    )
    key_cutoff: Final = (
        ', (SELECT "reset_at" FROM "LiteLLM_FleetKeyBudgetReset" WHERE "api_key" = $1)' if column == "api_key" else ""
    )
    timeout_ms: Final = max(1, int(PROXY_DB_LOOKUP_DEADLINE_SECONDS * 750))
    async with db_lookup_gate.current():
        if not await fleet_budget_replication_started(prisma_client):
            return 0.0
        async with prisma_client.writer_db.tx() as tx:
            await tx.execute_raw(f"SET LOCAL statement_timeout = {timeout_ms}")
            rows: Final[Sequence[Mapping[str, object]]] = await tx.query_raw(
                f'''SELECT COALESCE(SUM(spend), 0)::double precision AS spend
                    FROM "LiteLLM_FleetBudgetSpend"
                    WHERE source_region IS NOT NULL AND "{column}" = $1
                        AND "startTime" >= GREATEST('-infinity'::timestamp,
                            $2::text::timestamptz AT TIME ZONE 'UTC'{key_cutoff})''',
                replicated_budget_entity_id(entity_id),
                cutoff,
            )
    value: Final = rows[0].get("spend") if rows else None
    return float(value) if isinstance(value, int | float) else 0.0
