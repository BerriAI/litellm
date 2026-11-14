import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import chain
from typing import TYPE_CHECKING, Final, Protocol, cast  # noqa: TID251  # Prisma generates transaction types

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import LiteralString

from litellm.models.performance_alert import (
    BASELINE_DAYS,
    FREE_ALERT_LIMIT,
    AlertMonitoring,
    Claim,
    PerformanceScope,
    ResourceUnreadableError,
    ResourceWindow,
    SweepOwnershipLostError,
    SweepState,
    advance,
    detect,
)
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.repositories.alert_repository import record_alerts
from litellm.repositories.config_repository import ConfigRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

KEY_DATE_INDEX: Final = "LiteLLM_DailyUserSpend_api_key_date_idx"
_STATEMENT_TIMEOUT: Final = "2s"
_TRANSACTION_TIMEOUT: Final = timedelta(seconds=5)


class _Transaction(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...
    async def execute_raw(self, query: str, *args: object) -> int: ...


class _Database(Protocol):
    def tx(self, *, timeout: timedelta) -> AbstractAsyncContextManager[_Transaction]: ...


class _MonitoringRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    alerts_used: int | None


class _StateRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: date | None
    scope: PerformanceScope | None
    cursor: str
    wrapped: bool
    checked: int
    skipped: int
    completed: bool
    start_scope: PerformanceScope | None
    start_after: str


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


_SOURCES: Final[Mapping[PerformanceScope, _Source]] = {
    "gateway": _Source('"LiteLLM_DailyModelUsage"', "'gateway'", "request_count", ""),
    "model": _Source('"LiteLLM_DailyModelUsage"', "model_group", "request_count", ""),
    "team": _Source('"LiteLLM_DailyTeamSpend"', "team_id", "api_requests", ""),
    "user": _Source('"LiteLLM_DailyUserSpend"', "user_id", "api_requests", ""),
    "key": _Source(
        '"LiteLLM_DailyUserSpend"',
        "api_key",
        "api_requests",
        " AND api_key ~ '^[0-9a-f]{64}$'",
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
    return f"""SELECT {source.resource_id} AS resource_id,
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
    LIMIT $3::int"""


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
RETURNING day, scope, resume_after AS cursor, wrapped, checked, skipped, completed_at IS NOT NULL AS completed,
    start_scope, start_after"""
_SAVE: Final = """UPDATE "LiteLLM_PerformanceAlertState"
SET day = $2, scope = $3, resume_after = $4, wrapped = $5, checked = $6, skipped = $7,
    start_scope = $9, start_after = $10, completed_at = CASE WHEN $8 THEN now() AT TIME ZONE 'UTC' END, updated_at = now() AT TIME ZONE 'UTC'
WHERE id = 1 AND owner = $1"""
_INDEX_READY: Final = """SELECT 1 AS resource_id FROM pg_index WHERE indexrelid = to_regclass($1) AND indisvalid"""


class PerformanceAlertRepository:
    def __init__(self, prisma_client: "PrismaClient", *, enterprise: bool = False, configured: bool = True) -> None:
        self._database: Final = cast(  # cast-ok: generated Prisma transactions expose the bounded SQL surface
            _Database, prisma_client.writer_db
        )
        self._enterprise: Final = enterprise
        self._configured: Final = configured
        self._config: Final = ConfigRepository(prisma_client, use_writer=True)

    async def _monitoring(self, transaction: _Transaction, *, lock: bool = False) -> AlertMonitoring:
        query: Final = 'SELECT enabled, alerts_used FROM "LiteLLM_PerformanceAlertState" WHERE id = 1'
        rows: Final = await transaction.query_raw(query + (" FOR UPDATE" if lock else ""))
        if not rows:
            await transaction.execute_raw(
                """INSERT INTO "LiteLLM_PerformanceAlertState" (id, owner, next_run_at, updated_at)
                VALUES (1, '', now(), now() AT TIME ZONE 'UTC') ON CONFLICT (id) DO NOTHING"""
            )
            return await self._monitoring(transaction, lock=lock)
        row: Final = _MonitoringRow.model_validate(rows[0])
        if row.alerts_used is None:
            await transaction.execute_raw(
                """UPDATE "LiteLLM_PerformanceAlertState" SET alerts_used = (
                    SELECT COUNT(*) FROM (SELECT 1 FROM "LiteLLM_Alert"
                    WHERE type != 'model_deprecation' LIMIT $1) existing
                ) WHERE id = 1 AND alerts_used IS NULL""",
                FREE_ALERT_LIMIT,
            )
            return await self._monitoring(transaction, lock=lock)
        return AlertMonitoring(
            enabled=self._configured and row.enabled,
            alerts_used=row.alerts_used,
            alert_limit=None if self._enterprise else FREE_ALERT_LIMIT,
            disabled_by_config=not self._configured,
        )

    async def monitoring(self) -> AlertMonitoring:
        async with self._transaction() as transaction:
            monitoring: Final = await self._monitoring(transaction)
        return monitoring.model_copy(
            update={"model_deprecations_enabled": await self._config.model_deprecations_enabled()}
        )

    async def set_model_deprecations_enabled(self, enabled: bool) -> AlertMonitoring:
        await self._config.set_model_deprecations_enabled(enabled)
        return await self.monitoring()

    async def set_enabled(self, enabled: bool) -> AlertMonitoring:
        async with self._database.tx(timeout=timedelta(seconds=10)) as transaction:
            await transaction.query_raw(
                "SELECT set_config('statement_timeout', '6s', true), set_config('lock_timeout', '5s', true)"
            )
            await self._monitoring(transaction, lock=True)
            await transaction.execute_raw(
                """UPDATE "LiteLLM_PerformanceAlertState" SET enabled = $1,
                updated_at = now() AT TIME ZONE 'UTC' WHERE id = 1""",
                enabled,
            )
        return await self.monitoring()

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
            if not (await self._monitoring(transaction)).active:
                return None
            rows: Final = _STATE_ROWS.validate_python(
                await transaction.query_raw(_CLAIM, owner, f"{int(interval.total_seconds())} seconds")
            )
        if not rows:
            return None
        row: Final = rows[0]
        if row.day is None or row.scope is None:
            return Claim(owner, None)
        return Claim(owner, SweepState.model_validate(row.model_dump()))

    async def key_index_ready(self) -> bool:
        async with self._transaction() as transaction:
            return bool(await transaction.query_raw(_INDEX_READY, f'"{KEY_DATE_INDEX}"'))

    async def evaluate_page(self, owner: str, state: SweepState, limit: int) -> SweepState | None:
        """Evaluate the next `limit` resources and save their alerts together with the new position."""
        from prisma.errors import RawQueryError

        try:
            return await self._evaluate_page(owner, state, limit)
        except ValidationError as exc:
            raise ResourceUnreadableError() from exc
        except RawQueryError as exc:
            if PrismaDBExceptionHandler.is_statement_timeout_error(exc):
                raise ResourceUnreadableError() from exc
            raise

    async def _evaluate_page(self, owner: str, state: SweepState, limit: int) -> SweepState | None:
        async with self._transaction() as transaction:
            monitoring: Final = await self._monitoring(transaction, lock=True)
            if not monitoring.active:
                return None
            windows: Final = _WINDOWS.validate_python(
                await transaction.query_raw(_page_sql(state.scope, state.wrapped), *self._arguments(state, limit))
            )
            saved: Final = await record_alerts(
                transaction,
                tuple(chain.from_iterable(detect(state.scope, window, state.day) for window in windows)),
                None if self._enterprise else FREE_ALERT_LIMIT - monitoring.alerts_used,
            )
            if saved:
                await transaction.execute_raw(
                    """UPDATE "LiteLLM_PerformanceAlertState"
                    SET alerts_used = LEAST($1, alerts_used + $2) WHERE id = 1""",
                    FREE_ALERT_LIMIT,
                    saved,
                )
            if not self._enterprise and monitoring.alerts_used + saved >= FREE_ALERT_LIMIT:
                await self._save(transaction, owner, state)
                return None
            advanced: Final = advance(state, tuple(window.resource_id for window in windows), limit)
            await self._save(transaction, owner, advanced)
            return advanced

    async def skip_next(self, owner: str, state: SweepState) -> SweepState | None:
        """Step over the next resource without evaluating it, for one that cannot be read within the limits."""
        async with self._transaction() as transaction:
            if not (await self._monitoring(transaction, lock=True)).active:
                return None
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
            return (*window, state.cursor, state.start)
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
            state.start_scope,
            state.start_after,
        )
        if updated != 1:
            raise SweepOwnershipLostError()
