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
from litellm.proxy.db.queries.autorouter import (
    ADVANCE_BASELINE_COMPARISON_REVISION,
    APPLY_BASELINE_SESSION_CORRECTIONS,
    APPLY_BASELINE_USER_SESSION_CORRECTIONS,
    CLAIM_DIRTY_BASELINE_COMPARISONS,
    CREATE_BASELINE_COMPARISON,
    DELETE_RETIRED_BASELINE_OBSERVATIONS,
    FIND_BASELINE_OBSERVATION_CHANGED_BEFORE,
    FIND_BASELINE_OBSERVATION_WITHOUT_SPEND_LOG,
    INSERT_BASELINE_OBSERVATION,
    LOCK_BASELINE_COMPARISON,
    MARK_BASELINE_OBSERVATION_CONFLICTED,
    PUBLISH_BASELINE_COMPARISON_HISTORY,
    PUBLISH_BASELINE_TO_SPEND_LOGS,
    READ_BASELINE_OBSERVATION,
    READ_BASELINE_OBSERVATION_PAGE,
    RETIRE_EXPIRED_BASELINE_COMPARISONS,
    STORE_BASELINE_PUBLICATIONS,
)
from litellm.proxy.db.queries.transaction import SET_LOCK_TIMEOUT, SET_STATEMENT_TIMEOUT
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
    await db.execute_raw(PUBLISH_BASELINE_TO_SPEND_LOGS, serialized)
    await db.execute_raw(APPLY_BASELINE_SESSION_CORRECTIONS, serialized)
    if any(change.user_id for change in changes):
        await db.execute_raw(APPLY_BASELINE_USER_SESSION_CORRECTIONS, serialized)
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
    await db.execute_raw(STORE_BASELINE_PUBLICATIONS, serialized)


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
                await db.execute_raw(SET_STATEMENT_TIMEOUT, "5000")
                await db.execute_raw(SET_LOCK_TIMEOUT, "1000")
                await db.execute_raw(
                    CREATE_BASELINE_COMPARISON, record.scope, record.api_key, record.session_id, record.router_name
                )
                rows: Final = _COMPARISONS.validate_python(
                    tuple(await db.query_raw(LOCK_BASELINE_COMPARISON, record.scope))
                )
                if not rows:
                    return "unavailable"
                revision: Final = rows[0].revision + 1
                data: Final = _serialized(record)
                inserted: Final = await db.execute_raw(
                    INSERT_BASELINE_OBSERVATION,
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
                        MARK_BASELINE_OBSERVATION_CONFLICTED,
                        record.observation.request_id,
                        record.scope,
                        data,
                        revision,
                    )
                )
                canonical: Final = (
                    _RECORDS.validate_python(
                        tuple(
                            await db.query_raw(
                                READ_BASELINE_OBSERVATION,
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
                        ADVANCE_BASELINE_COMPARISON_REVISION,
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
        cursor: float | None = None  # rebind-ok: keyset pagination advances after each complete timestamp group
        while page := _RECORDS.validate_python(
            tuple(
                await db.query_raw(
                    READ_BASELINE_OBSERVATION_PAGE, scope, after_revision, cursor, _PAGE_TIMESTAMPS, withdraw_from
                )
            )
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
            await db.execute_raw(SET_STATEMENT_TIMEOUT, str(max(1, timeout_ms)))
            await db.execute_raw(SET_LOCK_TIMEOUT, str(max(1, timeout_ms)))
            await db.execute_raw(
                RETIRE_EXPIRED_BASELINE_COMPARISONS,
                cutoff,
                batch_size,
            )
            await db.execute_raw(
                DELETE_RETIRED_BASELINE_OBSERVATIONS,
                cutoff,
                batch_size,
            )

    async def project(self, scope: str) -> Literal["published", "unchanged", "unavailable"]:
        try:
            async with self.transaction() as db:
                await db.execute_raw(SET_STATEMENT_TIMEOUT, "5000")
                await db.execute_raw(SET_LOCK_TIMEOUT, "1000")
                rows: Final = _COMPARISONS.validate_python(tuple(await db.query_raw(LOCK_BASELINE_COMPARISON, scope)))
                if not rows or rows[0].retired or rows[0].revision == rows[0].published_revision:
                    return "unchanged"
                missing_log: Final = await db.query_raw(
                    FIND_BASELINE_OBSERVATION_WITHOUT_SPEND_LOG,
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
                    FIND_BASELINE_OBSERVATION_CHANGED_BEFORE,
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
                    PUBLISH_BASELINE_COMPARISON_HISTORY,
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
    except (Exception, asyncio.CancelledError) as error:
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
            await db.execute_raw(SET_STATEMENT_TIMEOUT, "1000")
            scopes: Final = _SCOPES.validate_python(tuple(await db.query_raw(CLAIM_DIRTY_BASELINE_COMPARISONS)))
        slots: Final = asyncio.Semaphore(4)

        async def project(item: _Scope) -> str:
            async with slots:
                return await store.project(item.scope)

        outcomes: Final = await asyncio.wait_for(asyncio.gather(*(project(item) for item in scopes)), timeout=5)
        if len(scopes) == 32 and "published" in outcomes:
            request_spend_log_flush(client)
    except Exception:  # noqa: BLE001  # durable dirty comparisons remain eligible after the retry interval
        verbose_proxy_logger.warning("Auto-router baseline projection will retry on a later spend flush")
