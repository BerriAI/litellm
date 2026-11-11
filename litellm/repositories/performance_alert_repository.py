import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from itertools import chain
from typing import TYPE_CHECKING, Final, Protocol, cast  # noqa: TID251  # Prisma generates transaction types

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import LiteralString

from litellm.models.performance_alert import (
    BASELINE_DAYS,
    Claim,
    PerformanceScope,
    ResourceUnreadableError,
    ResourceWindow,
    SweepOwnershipLostError,
    SweepState,
    advance,
    detect,
    rotation_start,
)
from litellm.repositories.alert_repository import record_alerts

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

KEY_DATE_INDEX: Final = "LiteLLM_DailyUserSpend_api_key_date_idx"
_STATEMENT_TIMED_OUT: Final = "`57014`"
_STATEMENT_TIMEOUT: Final = "2s"
_TRANSACTION_TIMEOUT: Final = timedelta(seconds=5)


class _Transaction(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...
    async def execute_raw(self, query: str, *args: object) -> int: ...


class _Database(Protocol):
    def tx(self, *, timeout: timedelta) -> AbstractAsyncContextManager[_Transaction]: ...


class _StateRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: date | None
    scope: PerformanceScope | None
    cursor: str
    wrapped: bool
    checked: int
    skipped: int
    completed: bool


class _IdRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    resource_id: str


_STATE_ROWS: Final = TypeAdapter(tuple[_StateRow, ...])
_WINDOWS: Final = TypeAdapter(tuple[ResourceWindow, ...])
_ID_ROWS: Final = TypeAdapter(tuple[_IdRow, ...])


@dataclass(frozen=True, slots=True)
class _Source:
    table: LiteralString
    resource_id: LiteralString
    requests: LiteralString
    filter: LiteralString
    exists: LiteralString


_SOURCES: Final[Mapping[PerformanceScope, _Source]] = {
    "gateway": _Source('"LiteLLM_DailyModelUsage"', "'gateway'", "request_count", "", "true"),
    "model": _Source('"LiteLLM_DailyModelUsage"', "model_group", "request_count", "", "true"),
    "team": _Source(
        '"LiteLLM_DailyTeamSpend"',
        "team_id",
        "api_requests",
        "",
        'EXISTS (SELECT 1 FROM "LiteLLM_TeamTable" t WHERE t.team_id = page.resource_id AND NOT t.blocked)',
    ),
    "user": _Source(
        '"LiteLLM_DailyUserSpend"',
        "user_id",
        "api_requests",
        "",
        'EXISTS (SELECT 1 FROM "LiteLLM_UserTable" u WHERE u.user_id = page.resource_id)',
    ),
    "key": _Source(
        '"LiteLLM_DailyUserSpend"',
        "api_key",
        "api_requests",
        " AND api_key ~ '^[0-9a-f]{64}$'",
        'EXISTS (SELECT 1 FROM "LiteLLM_VerificationToken" k WHERE k.token = page.resource_id'
        " AND k.blocked IS NOT TRUE AND (k.expires IS NULL OR k.expires > now()))",
    ),
}


def _bounds(source: _Source, wrapped: bool) -> LiteralString:
    upper: Final = f" AND {source.resource_id} <= $5" if wrapped else ""
    return (
        f" AND {source.resource_id} > $4{upper} AND {source.resource_id} <> ''"
        f" AND octet_length({source.resource_id}) <= 1024{source.filter}"
    )


def _page_sql(scope: PerformanceScope, wrapped: bool) -> LiteralString:
    source: Final = _SOURCES[scope]
    where: Final = "" if scope == "gateway" else _bounds(source, wrapped)
    group: Final = (
        " HAVING COUNT(*) > 0"
        if scope == "gateway"
        else f" GROUP BY {source.resource_id} ORDER BY {source.resource_id}"
    )
    return f"""WITH page AS (
    SELECT {source.resource_id} AS resource_id,
        COALESCE(SUM(spend) FILTER (WHERE date = $1), 0)::float8 AS spend,
        COALESCE(SUM(spend) FILTER (WHERE date < $1), 0)::float8 AS base_spend,
        COALESCE(SUM({source.requests}) FILTER (WHERE date = $1), 0)::bigint AS requests,
        COALESCE(SUM({source.requests}) FILTER (WHERE date < $1), 0)::bigint AS base_requests,
        COALESCE(SUM(failed_requests) FILTER (WHERE date = $1), 0)::bigint AS failed,
        COALESCE(SUM(failed_requests) FILTER (WHERE date < $1), 0)::bigint AS base_failed,
        COALESCE(SUM(total_response_time_ms) FILTER (WHERE date = $1), 0)::bigint AS response_ms,
        COALESCE(SUM(timed_requests) FILTER (WHERE date = $1), 0)::bigint AS timed,
        COALESCE(SUM(total_response_time_ms) FILTER (WHERE date < $1), 0)::bigint AS base_response_ms,
        COALESCE(SUM(timed_requests) FILTER (WHERE date < $1), 0)::bigint AS base_timed,
        COUNT(DISTINCT date) FILTER (WHERE date < $1 AND {source.requests} > 0) AS active_days,
        COUNT(DISTINCT date) FILTER (WHERE date < $1 AND timed_requests > 0) AS timed_days
    FROM {source.table}
    WHERE date BETWEEN $2 AND $1{where}{group}
    LIMIT $3::int
)
SELECT page.*, {source.exists} AS exists FROM page ORDER BY resource_id"""


def _next_id_sql(scope: PerformanceScope, wrapped: bool) -> LiteralString:
    source: Final = _SOURCES[scope]
    return f"""SELECT {source.resource_id} AS resource_id FROM {source.table}
WHERE date BETWEEN $2 AND $1{_bounds(source, wrapped)}
ORDER BY {source.resource_id} LIMIT $3::int"""


_LIMITS: Final = f"""SELECT set_config('statement_timeout', '{_STATEMENT_TIMEOUT}', true),
    set_config('lock_timeout', '100ms', true), set_config('work_mem', '1MB', true),
    set_config('max_parallel_workers_per_gather', '0', true)"""
_CLAIM: Final = """INSERT INTO "LiteLLM_PerformanceAlertState" (id, owner, next_run_at, updated_at)
VALUES (1, $1, now() + $2::interval, now() AT TIME ZONE 'UTC')
ON CONFLICT (id) DO UPDATE SET owner = EXCLUDED.owner, next_run_at = EXCLUDED.next_run_at,
    updated_at = now() AT TIME ZONE 'UTC'
WHERE "LiteLLM_PerformanceAlertState".next_run_at <= now() + interval '30 seconds'
RETURNING day, scope, resume_after AS cursor, wrapped, checked, skipped, completed_at IS NOT NULL AS completed"""
_SAVE: Final = """UPDATE "LiteLLM_PerformanceAlertState"
SET day = $2, scope = $3, resume_after = $4, wrapped = $5, checked = $6, skipped = $7,
    completed_at = CASE WHEN $8 THEN now() AT TIME ZONE 'UTC' END, updated_at = now() AT TIME ZONE 'UTC'
WHERE id = 1 AND owner = $1"""
_ACCOUNTING_PENDING: Final = """SELECT 1 AS resource_id FROM "LiteLLM_SpendLogDelivery"
WHERE state = 'pending' AND created_at < $1::timestamp LIMIT 1"""
_INDEX_READY: Final = """SELECT 1 AS resource_id FROM pg_index WHERE indexrelid = to_regclass($1) AND indisvalid"""


def is_statement_timeout(error: Exception) -> bool:
    return _STATEMENT_TIMED_OUT in str(error)


class PerformanceAlertRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._database: Final = cast(  # cast-ok: generated Prisma transactions expose the bounded SQL surface
            _Database, prisma_client.writer_db
        )

    @asynccontextmanager
    async def _transaction(self) -> AsyncGenerator[_Transaction]:
        async with self._database.tx(timeout=_TRANSACTION_TIMEOUT) as transaction:
            await transaction.query_raw(_LIMITS)
            yield transaction

    async def claim(self, interval: timedelta) -> Claim | None:
        """Take this deployment's next run, at most one per `interval` across every worker and pod. A worker whose
        timer fires up to 30 seconds early still takes it, so the claim's own delay never pushes a run back a whole
        interval."""
        owner: Final = uuid.uuid4().hex
        async with self._transaction() as transaction:
            rows: Final = _STATE_ROWS.validate_python(
                await transaction.query_raw(_CLAIM, owner, f"{int(interval.total_seconds())} seconds")
            )
        if not rows:
            return None
        row: Final = rows[0]
        if row.day is None or row.scope is None:
            return Claim(owner, None)
        return Claim(owner, SweepState.model_validate(row.model_dump()))

    async def accounting_pending(self, day: date) -> bool:
        day_end: Final = datetime.combine(day + timedelta(days=1), time.min, timezone.utc)
        async with self._transaction() as transaction:
            return bool(await transaction.query_raw(_ACCOUNTING_PENDING, day_end.replace(tzinfo=None).isoformat()))

    async def key_index_ready(self) -> bool:
        async with self._transaction() as transaction:
            return bool(await transaction.query_raw(_INDEX_READY, f'"{KEY_DATE_INDEX}"'))

    async def save(self, owner: str, state: SweepState) -> None:
        async with self._transaction() as transaction:
            await self._save(transaction, owner, state)

    async def evaluate_page(self, owner: str, state: SweepState, limit: int) -> SweepState:
        """Evaluate the next `limit` resources and save their alerts together with the new position."""
        from prisma.errors import RawQueryError

        try:
            return await self._evaluate_page(owner, state, limit)
        except ValidationError as exc:
            raise ResourceUnreadableError() from exc
        except RawQueryError as exc:
            if is_statement_timeout(exc):
                raise ResourceUnreadableError() from exc
            raise

    async def _evaluate_page(self, owner: str, state: SweepState, limit: int) -> SweepState:
        async with self._transaction() as transaction:
            windows: Final = _WINDOWS.validate_python(
                await transaction.query_raw(_page_sql(state.scope, state.wrapped), *self._arguments(state, limit))
            )
            await record_alerts(
                transaction, tuple(chain.from_iterable(detect(state.scope, window, state.day) for window in windows))
            )
            advanced: Final = advance(state, tuple(window.resource_id for window in windows), limit)
            await self._save(transaction, owner, advanced)
            return advanced

    async def skip_next(self, owner: str, state: SweepState) -> SweepState:
        """Step over the next resource without evaluating it, for one that cannot be read within the limits."""
        async with self._transaction() as transaction:
            rows: Final = (
                (_IdRow(resource_id="gateway"),)
                if state.scope == "gateway"
                else _ID_ROWS.validate_python(
                    await transaction.query_raw(_next_id_sql(state.scope, state.wrapped), *self._arguments(state, 1))
                )
            )
            advanced: Final = advance(state, tuple(row.resource_id for row in rows), 1, skipped=len(rows))
            await self._save(transaction, owner, advanced)
            return advanced

    @staticmethod
    def _arguments(state: SweepState, limit: int) -> tuple[str | int, ...]:
        start: Final = state.day - timedelta(days=BASELINE_DAYS)
        window: Final = (state.day.isoformat(), start.isoformat(), limit)
        if state.scope == "gateway":
            return window
        if state.wrapped:
            return (*window, state.cursor, rotation_start(state.day))
        return (*window, state.cursor)

    @staticmethod
    async def _save(transaction: _Transaction, owner: str, state: SweepState) -> None:
        updated: Final = await transaction.execute_raw(
            _SAVE,
            owner,
            state.day.isoformat(),
            state.scope,
            state.cursor,
            state.wrapped,
            state.checked,
            state.skipped,
            state.completed,
        )
        if updated != 1:
            raise SweepOwnershipLostError()
