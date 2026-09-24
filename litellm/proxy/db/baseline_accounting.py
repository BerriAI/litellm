from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime, timedelta
from functools import reduce
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from typing_extensions import Self

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.autorouter_session_rollup import (
    AutoRouterTurnTransaction,
    write_autorouter_turn,
)
from litellm.proxy.db.create_views import SupportsRawQueries
from litellm.proxy.db.daily_spend_bulk_upsert import (
    DAILY_SPEND_TABLES,
    DailySpendEntity,
    SpendRow,
    build_bulk_upsert,
    merge_by_conflict_key,
)
from litellm.proxy.db.routing_prisma_wrapper import writer_wrapper
from litellm.proxy.spend_tracking.baseline_accounting import (
    BaselineEstimate,
    BaselineHistory,
    BaselineObservation,
    advance_baseline_history,
)
from litellm.proxy.spend_tracking.savings import BaselineCosts, BaselineCostSnapshot, price_baseline_comparison

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class DailyBaselineTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    entity: DailySpendEntity
    entity_id: str | None


class DailyBaselineAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    date: str
    api_key: str
    model: str | None = None
    custom_llm_provider: str | None = None
    model_group: str | None = None
    endpoint: str | None = None
    mcp_namespaced_tool_name: str | None = None
    targets: tuple[DailyBaselineTarget, ...] = ()

    def adjustment(self, target: DailyBaselineTarget, savings_delta: float, request_id: str) -> SpendRow:
        table: Final = DAILY_SPEND_TABLES[target.entity]
        return MappingProxyType(
            {
                "date": self.date,
                "api_key": self.api_key,
                "model": self.model,
                "custom_llm_provider": self.custom_llm_provider,
                "model_group": self.model_group,
                "endpoint": self.endpoint,
                "mcp_namespaced_tool_name": self.mcp_namespaced_tool_name,
                table.entity_id_column: target.entity_id,
                "request_id": request_id,
                "autorouter_savings_spend": savings_delta,
            }
        )


class BaselineAccountingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scope: str = Field(pattern=r"^autorouter-baseline:v3:[a-f0-9]{64}$")
    api_key: str = Field(min_length=1)
    session_id: str = Field(min_length=1, max_length=256)
    router_name: str = Field(min_length=1)
    baseline_model: str = Field(min_length=1)
    observation: BaselineObservation
    pricing: BaselineCostSnapshot
    turn: AutoRouterTurnTransaction | None
    daily: DailyBaselineAttribution | None

    @model_validator(mode="after")
    def consistent_turn(self) -> Self:
        turn: Final = self.turn
        if turn is not None and (
            (turn.api_key, turn.session_id, turn.router_name, turn.baseline_model)
            != (self.api_key, self.session_id, self.router_name, self.baseline_model)
            or turn.spend != self.pricing.actual_spend + self.pricing.classifier_cost
            or any(
                (
                    turn.saved_spend,
                    turn.savings_estimated_turns,
                    turn.savings_estimated_actual_spend,
                    turn.savings_estimated_saved_spend,
                )
            )
        ):
            raise ValueError("Baseline observation must own an unestimated turn with matching scope and actual cost")
        return self


class BaselinePublication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[3] = 3
    comparison_id: str
    comparison_started_at: float
    status: Literal["estimated", "unknown"]
    reason: str
    provenance: Literal["observed_identical", "modeled"] | None = None
    actual_spend: float | None = None
    baseline_spend: float | None = None
    input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_5m_input_tokens: int | None = None
    cache_creation_1h_input_tokens: int | None = None

    @property
    def costs(self) -> BaselineCosts | None:
        if self.status != "estimated" or self.actual_spend is None or self.baseline_spend is None:
            return None
        return BaselineCosts(self.actual_spend, self.baseline_spend)


def baseline_publication(
    record: BaselineAccountingRecord, estimate: BaselineEstimate, first_at: float
) -> BaselinePublication:
    costs: Final = price_baseline_comparison(record.pricing, estimate.usage, estimate.provenance)
    details: Final = estimate.usage.prompt_tokens_details if estimate.usage is not None else None
    writes: Final = details.cache_creation_token_details if details is not None else None
    return BaselinePublication(
        comparison_id=record.scope,
        comparison_started_at=first_at,
        status="estimated" if costs is not None else "unknown",
        reason=estimate.reason if costs is not None or estimate.usage is None else "pricing_unavailable",
        provenance=estimate.provenance if costs is not None else None,
        actual_spend=costs.actual if costs is not None else None,
        baseline_spend=costs.baseline if costs is not None else None,
        input_tokens=details.text_tokens if details is not None else None,
        cache_read_input_tokens=details.cached_tokens if details is not None else None,
        cache_creation_5m_input_tokens=writes.ephemeral_5m_input_tokens if writes is not None else None,
        cache_creation_1h_input_tokens=writes.ephemeral_1h_input_tokens if writes is not None else None,
    )


class _Comparison(BaseModel):
    revision: int
    published_revision: int
    initial_equivalent: bool
    retired: bool
    history: str | None


class _StoredRecord(BaseModel):
    data: str
    publication: str | None
    conflicted: bool
    started_at: float


class _Change(BaseModel):
    request_id: str
    publication: BaselinePublication
    api_key: str
    user_id: str = ""
    session_id: str
    router_name: str
    baseline_model: str
    covered_delta: int
    actual_delta: float
    savings_delta: float
    daily: DailyBaselineAttribution | None


class _TransactionManager(Protocol):
    async def __aenter__(self) -> SupportsRawQueries: ...

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


class _TransactionalDatabase(Protocol):
    def tx(self, *, timeout: timedelta) -> _TransactionManager: ...


_COMPARISONS: Final = TypeAdapter(tuple[_Comparison, ...])
_RECORDS: Final = TypeAdapter(tuple[_StoredRecord, ...])
_HISTORY: Final = TypeAdapter(BaselineHistory)
_PAGE_TIMESTAMPS: Final = 128
_TRANSACTION_TIMEOUT: Final = timedelta(seconds=10)

_CREATE_COMPARISON: Final = """
INSERT INTO "LiteLLM_AutoRouterBaselineComparison"
    (scope, api_key, session_id, router_name, initial_equivalent)
VALUES ($1, $2, $3, $4, NOT EXISTS (
    SELECT 1 FROM "LiteLLM_AutoRouterSession"
    WHERE api_key = $2 AND session_id = $3 AND router_name = $4
)) ON CONFLICT (scope) DO NOTHING
"""
_LOCK_COMPARISON: Final = """
SELECT revision, published_revision, initial_equivalent, retired, history
FROM "LiteLLM_AutoRouterBaselineComparison" WHERE scope = $1 FOR UPDATE
"""
_INSERT_RECORD: Final = """
INSERT INTO "LiteLLM_AutoRouterBaselineObservation"
    (request_id, scope, started_at, revision, data)
VALUES ($1, $2, $3::float8, $4::bigint, $5)
ON CONFLICT (request_id) DO NOTHING
"""
_MARK_CONFLICT: Final = """
UPDATE "LiteLLM_AutoRouterBaselineObservation"
SET conflicted = TRUE, revision = $4::bigint
WHERE request_id = $1 AND scope = $2 AND data <> $3 AND NOT conflicted
"""
_READ_PAGE: Final = """
WITH times AS (
    SELECT DISTINCT started_at FROM "LiteLLM_AutoRouterBaselineObservation"
    WHERE scope = $1 AND revision > $2::bigint
      AND ($3::float8 IS NULL OR started_at > $3::float8)
      AND ($5::float8 IS NULL OR (
          started_at >= $5::float8 AND publication::jsonb->>'status' = 'estimated'
      ))
    ORDER BY started_at LIMIT $4::int
)
SELECT data, publication, conflicted, started_at
FROM "LiteLLM_AutoRouterBaselineObservation"
WHERE scope = $1 AND revision > $2::bigint
  AND started_at IN (SELECT started_at FROM times)
  AND ($5::float8 IS NULL OR publication::jsonb->>'status' = 'estimated')
ORDER BY started_at, request_id
"""
_UPDATE_LOGS: Final = """
WITH changes AS (
    SELECT request_id, publication::jsonb AS publication
    FROM jsonb_to_recordset($1::jsonb) AS x(request_id text, publication jsonb)
)
UPDATE "LiteLLM_SpendLogs" AS logs
SET metadata = (COALESCE(logs.metadata::jsonb, '{}'::jsonb) - 'autorouter_baseline_observation') || jsonb_build_object(
    'autorouter_savings_estimate', changes.publication,
    'autorouter_savings', CASE WHEN changes.publication->>'status' = 'estimated' THEN
        (changes.publication->>'baseline_spend')::float8 - (changes.publication->>'actual_spend')::float8
        ELSE NULL END
)
FROM changes WHERE logs.request_id = changes.request_id
"""
_UPDATE_PUBLICATIONS: Final = """
UPDATE "LiteLLM_AutoRouterBaselineObservation" AS observations
SET publication = x.publication::text
FROM jsonb_to_recordset($1::jsonb) AS x(request_id text, publication jsonb)
WHERE observations.request_id = x.request_id
"""


def _session_correction_sql(*, user_scoped: bool) -> str:
    table_name: Final = "LiteLLM_AutoRouterUserSession" if user_scoped else "LiteLLM_AutoRouterSession"
    identity_columns: Final = ("user_id, " if user_scoped else "") + "api_key, session_id, router_name"
    user_filter: Final = "WHERE user_id <> ''" if user_scoped else ""
    user_match: Final = "session.user_id = totals.user_id AND " if user_scoped else ""
    return f"""
WITH changes AS (
    SELECT * FROM jsonb_to_recordset($1::jsonb) AS x(
        user_id text, api_key text, session_id text, router_name text, baseline_model text,
        covered_delta int, actual_delta float8, savings_delta float8
    )
    {user_filter}
), totals AS (
    SELECT {identity_columns}, SUM(covered_delta)::int AS covered_delta,
        SUM(actual_delta) AS actual_delta, SUM(savings_delta) AS savings_delta
    FROM changes GROUP BY {identity_columns}
), models AS (
    SELECT {identity_columns}, jsonb_object_agg(baseline_model, delta) AS deltas
    FROM (
        SELECT {identity_columns}, baseline_model, SUM(covered_delta)::int AS delta
        FROM changes GROUP BY {identity_columns}, baseline_model
    ) grouped GROUP BY {identity_columns}
)
UPDATE "{table_name}" AS session
SET saved_spend = session.saved_spend + totals.savings_delta,
    savings_estimated_turns = session.savings_estimated_turns + totals.covered_delta,
    savings_estimated_actual_spend = session.savings_estimated_actual_spend + totals.actual_delta,
    savings_estimated_saved_spend = session.savings_estimated_saved_spend + totals.savings_delta,
    savings_estimated_baseline_models = (
        SELECT COALESCE(jsonb_object_agg(key, value), '{{}}'::jsonb) FROM (
            SELECT key, SUM(value::int)::int AS value FROM (
                SELECT * FROM jsonb_each_text(session.savings_estimated_baseline_models)
                UNION ALL SELECT * FROM jsonb_each_text(models.deltas)
            ) combined GROUP BY key HAVING SUM(value::int) > 0
        ) counts
    )
FROM totals JOIN models USING ({identity_columns})
WHERE {user_match}session.api_key = totals.api_key AND session.session_id = totals.session_id
    AND session.router_name = totals.router_name
"""


_UPDATE_SESSIONS: Final = _session_correction_sql(user_scoped=False)
_UPDATE_USER_SESSIONS: Final = _session_correction_sql(user_scoped=True)


def _primary_transaction(client: PrismaClient) -> _TransactionManager:
    primary: Final = cast(_TransactionalDatabase, writer_wrapper(client.db))
    return primary.tx(timeout=_TRANSACTION_TIMEOUT)


def _serialized(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _change(record: BaselineAccountingRecord, old: BaselinePublication | None, new: BaselinePublication) -> _Change:
    previous: Final = old.costs if old is not None else None
    current: Final = new.costs
    return _Change(
        request_id=record.observation.request_id,
        publication=new,
        api_key=record.api_key,
        user_id=record.turn.user_id if record.turn is not None else "",
        session_id=record.session_id,
        router_name=record.router_name,
        baseline_model=record.baseline_model,
        covered_delta=int(current is not None) - int(previous is not None),
        actual_delta=(current.actual if current is not None else 0.0)
        - (previous.actual if previous is not None else 0.0),
        savings_delta=(current.savings if current is not None else 0.0)
        - (previous.savings if previous is not None else 0.0),
        daily=record.daily,
    )


def _project_group(
    previous: tuple[BaselineHistory, tuple[_Change, ...]], stored: Sequence[_StoredRecord]
) -> tuple[BaselineHistory, tuple[_Change, ...]]:
    history, prior_changes = previous
    records: Final = tuple(BaselineAccountingRecord.model_validate_json(item.data) for item in stored)
    observations: Final = tuple(
        record.observation.model_copy(
            update=MappingProxyType(
                {"outcome": "uncertain", "baseline_equivalent": False, "reason": "conflicting_observation"}
            )
        )
        if row.conflicted
        else record.observation
        for record, row in zip(records, stored)
    )
    advanced, estimates = advance_baseline_history(history, observations)
    publications: Final = tuple(
        baseline_publication(
            record, estimate, advanced.first_at if advanced.first_at is not None else observations[0].started_at
        )
        for record, estimate in zip(records, estimates)
    )
    changes: Final = tuple(
        _change(record, old, publication)
        for record, row, publication in zip(records, stored, publications)
        for old in (BaselinePublication.model_validate_json(row.publication) if row.publication else None,)
        if publication != old
    )
    return advanced, (*prior_changes, *changes)


async def _publish(db: SupportsRawQueries, changes: Sequence[_Change]) -> None:
    if not changes:
        return
    serialized: Final = json.dumps(tuple(change.model_dump(mode="json") for change in changes), separators=(",", ":"))
    await db.execute_raw(_UPDATE_LOGS, serialized)
    await db.execute_raw(_UPDATE_SESSIONS, serialized)
    if any(change.user_id for change in changes):
        await db.execute_raw(_UPDATE_USER_SESSIONS, serialized)
    for entity, table in DAILY_SPEND_TABLES.items():
        if adjustments := tuple(
            change.daily.adjustment(target, change.savings_delta, change.request_id)
            for change in changes
            if change.daily is not None and change.savings_delta != 0
            for target in change.daily.targets
            if target.entity == entity
        ):
            statement, values = build_bulk_upsert(table, merge_by_conflict_key(table, adjustments))
            await db.execute_raw(statement, *values)
    await db.execute_raw(_UPDATE_PUBLICATIONS, serialized)


class BaselineAccountingStore:
    def __init__(self, transaction: Callable[[], _TransactionManager]) -> None:
        self.transaction: Final = transaction

    @classmethod
    def for_client(cls, client: PrismaClient) -> BaselineAccountingStore:
        def transaction() -> _TransactionManager:
            return _primary_transaction(client)

        return cls(transaction)

    async def append(
        self, record: BaselineAccountingRecord
    ) -> Literal["recorded", "retired", "conflict", "unavailable"]:
        try:
            async with self.transaction() as db:
                await db.execute_raw("SET LOCAL statement_timeout = 5000")
                await db.execute_raw("SET LOCAL lock_timeout = 1000")
                await db.execute_raw(
                    _CREATE_COMPARISON, record.scope, record.api_key, record.session_id, record.router_name
                )
                rows: Final = _COMPARISONS.validate_python(tuple(await db.query_raw(_LOCK_COMPARISON, record.scope)))
                if not rows:
                    return "unavailable"
                revision: Final = rows[0].revision + 1
                data: Final = _serialized(record)
                inserted: Final = await db.execute_raw(
                    _INSERT_RECORD,
                    record.observation.request_id,
                    record.scope,
                    record.observation.started_at,
                    revision,
                    data,
                )
                if inserted and record.turn is not None:
                    await write_autorouter_turn(db, record.turn)
                conflicted: Final = (
                    0
                    if inserted
                    else await db.execute_raw(
                        _MARK_CONFLICT, record.observation.request_id, record.scope, data, revision
                    )
                )
                canonical: Final = (
                    _RECORDS.validate_python(
                        tuple(
                            await db.query_raw(
                                'SELECT data, publication, conflicted, started_at FROM "LiteLLM_AutoRouterBaselineObservation" '
                                "WHERE request_id=$1 AND scope=$2",
                                record.observation.request_id,
                                record.scope,
                            )
                        )
                    )
                    if not inserted
                    else ()
                )
                if not inserted and not canonical:
                    return "conflict"
                if rows[0].retired:
                    await _publish(
                        db,
                        (
                            _change(
                                BaselineAccountingRecord.model_validate_json(canonical[0].data)
                                if canonical
                                else record,
                                BaselinePublication.model_validate_json(canonical[0].publication)
                                if canonical and canonical[0].publication is not None
                                else None,
                                BaselinePublication(
                                    comparison_id=record.scope,
                                    comparison_started_at=canonical[0].started_at
                                    if canonical
                                    else record.observation.started_at,
                                    status="unknown",
                                    reason="comparison_retired",
                                ),
                            ),
                        ),
                    )
                    return "retired"
                if inserted or conflicted:
                    await self._withdraw(
                        db, record.scope, canonical[0].started_at if canonical else record.observation.started_at
                    )
                    await db.execute_raw(
                        'UPDATE "LiteLLM_AutoRouterBaselineComparison" SET revision = $2::bigint, '
                        "updated_at = CURRENT_TIMESTAMP, attempted_at = NULL WHERE scope = $1",
                        record.scope,
                        revision,
                    )
            return "recorded"
        except Exception:  # noqa: BLE001  # accounting failure must not change inference or actual billing
            verbose_proxy_logger.warning("Auto-router baseline observation could not be persisted")
            return "unavailable"

    async def _pages(
        self, db: SupportsRawQueries, scope: str, after_revision: int, withdraw_from: float | None = None
    ) -> AsyncIterator[tuple[_StoredRecord, ...]]:
        cursor: float | None = None
        while page := _RECORDS.validate_python(
            tuple(await db.query_raw(_READ_PAGE, scope, after_revision, cursor, _PAGE_TIMESTAMPS, withdraw_from))
        ):
            yield page
            cursor = page[-1].started_at

    async def _withdraw(self, db: SupportsRawQueries, scope: str, started_at: float) -> None:
        async for page in self._pages(db, scope, 0, withdraw_from=started_at):
            await _publish(
                db,
                tuple(
                    _change(
                        BaselineAccountingRecord.model_validate_json(row.data),
                        previous,
                        BaselinePublication(
                            comparison_id=scope,
                            comparison_started_at=min(previous.comparison_started_at, started_at),
                            status="unknown",
                            reason="pending_projection",
                        ),
                    )
                    for row in page
                    if row.publication is not None
                    for previous in (BaselinePublication.model_validate_json(row.publication),)
                ),
            )

    async def retire_before(self, cutoff: datetime, batch_size: int, timeout_ms: int) -> None:
        async with self.transaction() as db:
            await db.execute_raw(f"SET LOCAL statement_timeout = {max(1, timeout_ms)}")
            await db.execute_raw(f"SET LOCAL lock_timeout = {max(1, timeout_ms)}")
            await db.execute_raw(
                'WITH expired AS (SELECT scope FROM "LiteLLM_AutoRouterBaselineComparison" '
                "WHERE NOT retired AND updated_at < $1::timestamptz ORDER BY updated_at "
                "LIMIT $2::int FOR UPDATE SKIP LOCKED) "
                'UPDATE "LiteLLM_AutoRouterBaselineComparison" AS comparison '
                "SET retired=TRUE, history=NULL FROM expired WHERE comparison.scope=expired.scope",
                cutoff,
                batch_size,
            )
            await db.execute_raw(
                'DELETE FROM "LiteLLM_AutoRouterBaselineObservation" WHERE request_id IN ('
                'SELECT event.request_id FROM "LiteLLM_AutoRouterBaselineObservation" AS event '
                'JOIN "LiteLLM_AutoRouterBaselineComparison" AS comparison USING (scope) '
                "WHERE comparison.retired AND comparison.updated_at < $1::timestamptz "
                "LIMIT $2::int)",
                cutoff,
                batch_size,
            )

    async def project(self, scope: str) -> Literal["published", "unchanged", "unavailable"]:
        try:
            async with self.transaction() as db:
                await db.execute_raw("SET LOCAL statement_timeout = 5000")
                await db.execute_raw("SET LOCAL lock_timeout = 1000")
                rows: Final = _COMPARISONS.validate_python(tuple(await db.query_raw(_LOCK_COMPARISON, scope)))
                if not rows or rows[0].retired or rows[0].revision == rows[0].published_revision:
                    return "unchanged"
                missing_log: Final = await db.query_raw(
                    'SELECT 1 FROM "LiteLLM_AutoRouterBaselineObservation" AS observation '
                    'WHERE scope=$1 AND publication IS NULL AND NOT EXISTS (SELECT 1 FROM "LiteLLM_SpendLogs" AS log '
                    "WHERE log.request_id=observation.request_id) LIMIT 1",
                    scope,
                )
                if missing_log:
                    return "unavailable"
                state: Final = rows[0]
                checkpoint: Final = (
                    _HISTORY.validate_json(state.history)
                    if state.history is not None
                    else BaselineHistory(equivalent=state.initial_equivalent)
                )
                changed: Final = await db.query_raw(
                    'SELECT 1 FROM "LiteLLM_AutoRouterBaselineObservation" '
                    "WHERE scope = $1 AND revision > $2::bigint AND started_at <= $3::float8 LIMIT 1",
                    scope,
                    state.published_revision,
                    checkpoint.last_at,
                )
                history = BaselineHistory(equivalent=state.initial_equivalent) if changed else checkpoint
                async for page in self._pages(db, scope, 0 if changed else state.published_revision):
                    history, updates = reduce(
                        _project_group,
                        (tuple(group) for _, group in groupby(page, key=lambda item: item.started_at)),
                        (history, ()),
                    )
                    await _publish(db, updates)
                await db.execute_raw(
                    'UPDATE "LiteLLM_AutoRouterBaselineComparison" '
                    "SET published_revision = revision, history = $2 WHERE scope = $1",
                    scope,
                    _HISTORY.dump_json(history).decode(),
                )
            return "published"
        except Exception:  # noqa: BLE001  # rollback leaves the durable revision dirty for a later flush
            verbose_proxy_logger.warning("Auto-router baseline projection remains pending")
            return "unavailable"


class _Scope(BaseModel):
    scope: str


_SCOPES: Final = TypeAdapter(tuple[_Scope, ...])
_CLAIM_DIRTY: Final = """
WITH candidates AS (
    SELECT scope FROM "LiteLLM_AutoRouterBaselineComparison"
    WHERE NOT retired AND revision <> published_revision
      AND (attempted_at IS NULL OR attempted_at < CURRENT_TIMESTAMP - INTERVAL '30 seconds')
    ORDER BY attempted_at NULLS FIRST, updated_at, scope LIMIT 32 FOR UPDATE SKIP LOCKED
)
UPDATE "LiteLLM_AutoRouterBaselineComparison" AS comparison
SET attempted_at = CURRENT_TIMESTAMP FROM candidates
WHERE comparison.scope = candidates.scope RETURNING comparison.scope
"""


async def _flush_records(
    store: BaselineAccountingStore, records: Sequence[BaselineAccountingRecord]
) -> tuple[BaselineAccountingRecord, ...]:
    slots: Final = asyncio.Semaphore(4)

    async def append(record: BaselineAccountingRecord) -> bool:
        async with slots:
            return await store.append(record) == "unavailable"

    failed: Final = await asyncio.gather(*(append(record) for record in records))
    return tuple(record for record, retry in zip(records, failed) if retry)


async def flush_baseline_accounting(client: PrismaClient) -> None:
    from litellm.proxy.utils import request_spend_log_flush

    store: Final = BaselineAccountingStore.for_client(client)
    async with client.baseline_accounting_lock:
        batch: Final = tuple(client.baseline_accounting_transactions[:32])
        client.baseline_accounting_transactions = client.baseline_accounting_transactions[32:]
        more_queued: Final = bool(client.baseline_accounting_transactions)
    try:
        remaining: Final = await asyncio.wait_for(_flush_records(store, batch), timeout=5)
    except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001  # unknown acknowledgements can be replayed safely
        async with client.baseline_accounting_lock:
            client.baseline_accounting_transactions.extend(batch)
        if isinstance(error, asyncio.CancelledError):
            raise
        return
    async with client.baseline_accounting_lock:
        client.baseline_accounting_transactions.extend(remaining)
    if more_queued and len(remaining) < len(batch):
        request_spend_log_flush(client)
    try:
        async with store.transaction() as db:
            await db.execute_raw("SET LOCAL statement_timeout = 1000")
            scopes: Final = _SCOPES.validate_python(tuple(await db.query_raw(_CLAIM_DIRTY)))
        slots: Final = asyncio.Semaphore(4)

        async def project(item: _Scope) -> str:
            async with slots:
                return await store.project(item.scope)

        outcomes: Final = await asyncio.wait_for(asyncio.gather(*(project(item) for item in scopes)), timeout=5)
        if len(scopes) == 32 and "published" in outcomes:
            request_spend_log_flush(client)
    except Exception:  # noqa: BLE001  # durable dirty comparisons remain eligible after the retry interval
        verbose_proxy_logger.warning("Auto-router baseline projection will retry on a later spend flush")
