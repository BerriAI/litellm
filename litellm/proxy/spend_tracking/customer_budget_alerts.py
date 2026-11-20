import asyncio
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, InstanceOf, TypeAdapter

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.duration_parser import get_budget_window_end
from litellm.proxy.db.budget_window_spend_writer import CommittedWindowSpend
from litellm.proxy.management_helpers.apps import AppStore, app_scope_identity
from litellm.proxy.management_helpers.customer_hierarchy import (
    PrismaRecord,
    customer_row_reset_at,
    effective_member_budget,
    related_budget,
    utc_now,
)
from litellm.types.integrations.customer_budget import (
    AppBudgetScope,
    CustomerBudgetEvent,
    CustomerBudgetEventType,
    CustomerBudgetScope,
)
from litellm.types.integrations.slack_alerting import SlackAlertingArgsEnum
from litellm.types.proxy.auth.customer_hierarchy import app_budget_limits
from litellm.types.proxy.management_endpoints.apps import App, AppFailure, AppMember, AppOrganization
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, CustomerBudgetUsage


class CustomerBudgetClaims(Protocol):
    async def async_increment_cache(self, *, key: str, value: float, ttl: float) -> float | None: ...


@runtime_checkable
class _CustomerBudgetTable(Protocol):
    async def find_unique(
        self, *, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> PrismaRecord | None: ...


class _CustomerRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    spend: float = 0.0
    alias: str | None = None
    customer_org_id: str | None = None
    customer_team_id: str | None = None
    budget: Mapping[str, object] | None = None
    budget_override: Mapping[str, object] | None = None
    budget_override_expires_at: datetime | None = None
    parent_org: "_CustomerRow | None" = None


@dataclass(frozen=True, slots=True)
class _Level:
    scope: CustomerBudgetScope
    transactions: str
    id_column: str
    table: str
    include: Mapping[str, object]


_LEVELS: Final = (
    _Level(
        "customer_org",
        "customer_org_list_transactions",
        "customer_org_id",
        "litellm_customerorganizationtable",
        {"budget": True},
    ),
    _Level(
        "customer_team",
        "customer_team_list_transactions",
        "customer_team_id",
        "litellm_customerteamtable",
        {"budget": True, "budget_override": True, "parent_org": True},
    ),
    _Level(
        "customer_member",
        "customer_member_list_transactions",
        "customer_member_id",
        "litellm_customermembertable",
        {"budget": True, "budget_override": True, "parent_org": True},
    ),
)
_CHARGES: Final = TypeAdapter(Mapping[str, float])
_TABLE: Final = TypeAdapter(InstanceOf[_CustomerBudgetTable])
_APP_SCOPES: Final[tuple[AppBudgetScope, ...]] = ("app", "app_customer_org", "app_customer_member")


def _events(level: _Level, entity_id: str, row: _CustomerRow, now: datetime) -> tuple[CustomerBudgetEvent, ...]:
    template: Final = related_budget(row.budget, now)
    override: Final = related_budget(row.budget_override, now, template.max_budget if template is not None else None)
    budget: Final = (
        effective_member_budget(template, override, row.budget_override_expires_at, now)
        if level.scope != "customer_org"
        else template
    )
    if budget is None:
        return ()
    customer_team: Final = row if level.scope == "customer_team" else None
    customer_org: Final = row if level.scope == "customer_org" else row.parent_org
    reset_at: Final = customer_row_reset_at(row, now)
    thresholds: Final[tuple[tuple[CustomerBudgetEventType, float | None], ...]] = (
        ("soft_budget_crossed", budget.soft_budget),
        ("budget_crossed", budget.max_budget),
    )
    return tuple(
        CustomerBudgetEvent(
            event=event,
            scope=level.scope,
            entity_id=entity_id,
            customer_org_id=customer_org.customer_org_id if customer_org else None,
            customer_org_alias=customer_org.alias if customer_org else None,
            customer_team_id=(
                customer_team.customer_team_id if level.scope != "customer_org" and customer_team else None
            ),
            customer_team_alias=customer_team.alias if level.scope != "customer_org" and customer_team else None,
            customer_member_id=entity_id if level.scope == "customer_member" else None,
            customer_member_alias=row.alias if level.scope == "customer_member" else None,
            spend=row.spend,
            budget=threshold,
            period=reset_at.isoformat() if reset_at is not None else "lifetime",
            timestamp=now,
        )
        for event, threshold in thresholds
        if threshold is not None and row.spend >= threshold
    )


async def _detect_customer_events(
    level: _Level, entity_id: str, database: object, cache: CustomerBudgetClaims, now: datetime
) -> None:
    try:
        table: Final = _TABLE.validate_python(getattr(getattr(database, "writer_db", database), level.table))
        record: Final = await table.find_unique(where={level.id_column: entity_id}, include=level.include)
        if record is None:
            return
        for event in _events(level, entity_id, _CustomerRow.model_validate(record.model_dump()), now):
            await CustomerBudgetAlerting.emit(event, cache)
    except Exception:  # noqa: BLE001  # Notification failures must not replay committed charges.
        verbose_proxy_logger.exception("Failed to send %s budget alert for %s", level.scope, entity_id)


async def _app_entity(
    store: AppStore, app: App, scope: AppBudgetScope, customer_id: str
) -> App | AppOrganization | AppMember | AppFailure | None:
    match scope:
        case "app":
            return app
        case "app_customer_org":
            return await store.find_organization(app, customer_id)
        case "app_customer_member":
            return await store.get_member(app.app_id, customer_id, known_to_app_only=True)


def _app_usages(
    budget: Budget,
    lifetime_spend: float,
    store: AppStore,
    windows: Sequence[CommittedWindowSpend] | None,
) -> tuple[CustomerBudgetUsage, ...]:
    if windows is None:
        return (
            (
                CustomerBudgetUsage(
                    source="scalar", budget_duration=None, max_budget=budget.max_budget, spend=lifetime_spend
                ),
            )
            if budget.max_budget is not None and budget.budget_duration is None
            else ()
        )
    return tuple(
        CustomerBudgetUsage(
            source="window",
            budget_duration=window.window_duration,
            max_budget=limit.max_budget,
            spend=window.spend,
            window_start=window.window_start,
            reset_at=window.reset_at
            or get_budget_window_end(
                window.window_duration, window.window_start, store.customers.settings.timezone, limit.reset_at
            ),
        )
        for window, limit in product(windows, app_budget_limits(budget))
        if window.window_duration == limit.budget_duration
    )


async def _app_events(
    scope: AppBudgetScope,
    entity_id: str,
    store: AppStore,
    windows: Sequence[CommittedWindowSpend] | None,
) -> tuple[CustomerBudgetEvent, ...]:
    app_id, customer_id = (entity_id, "") if scope == "app" else app_scope_identity(entity_id)
    app: Final = await store.get(app_id)
    if isinstance(app, AppFailure):
        return ()
    entity: Final = await _app_entity(store, app, scope, customer_id)
    if entity is None or isinstance(entity, AppFailure):
        return ()
    budget: Final = entity.budget if isinstance(entity, App) else entity.effective_budget
    if budget is None:
        return ()
    usages: Final = _app_usages(budget, entity.spend, store, windows)

    def event(kind: CustomerBudgetEventType, usage: CustomerBudgetUsage, threshold: float) -> CustomerBudgetEvent:
        return CustomerBudgetEvent(
            event=kind,
            scope=scope,
            entity_id=entity_id,
            app_id=app.app_id,
            app_alias=app.alias,
            customer_org_id=entity.customer_org_id if isinstance(entity, AppOrganization) else None,
            customer_member_id=entity.customer_member_id if isinstance(entity, AppMember) else None,
            spend=usage.spend or 0.0,
            budget=threshold,
            budget_duration=usage.budget_duration,
            period=usage.reset_at.isoformat() if usage.reset_at is not None else "lifetime",
            timestamp=store.now,
        )

    hard: Final = tuple(
        event("budget_crossed", usage, usage.max_budget)
        for usage in usages
        if usage.spend is not None and usage.spend >= usage.max_budget
    )
    if budget.soft_budget is None:
        return hard
    existing: Final = tuple(usage for usage in usages if usage.budget_duration == budget.budget_duration)
    soft_usages: Final = (
        existing
        if existing
        else _app_usages(
            Budget(
                max_budget=budget.soft_budget,
                budget_duration=budget.budget_duration,
                budget_limits=budget.budget_limits,
            ),
            entity.spend,
            store,
            windows,
        )
    )
    soft: Final = tuple(
        event("soft_budget_crossed", usage, budget.soft_budget)
        for usage in soft_usages
        if usage.budget_duration == budget.budget_duration
        and usage.spend is not None
        and usage.spend >= budget.soft_budget
    )
    return (*soft, *hard)


async def _detect_app_events(
    scope: AppBudgetScope,
    entity_id: str,
    database: object,
    cache: CustomerBudgetClaims,
    now: datetime,
    windows: Sequence[CommittedWindowSpend] | None = None,
) -> None:
    try:
        store: Final = AppStore(getattr(database, "writer_db", getattr(database, "db", database)), now)
        for event in await _app_events(scope, entity_id, store, windows):
            await CustomerBudgetAlerting.emit(event, cache)
    except Exception:  # noqa: BLE001  # Alert failures must not replay committed spend.
        verbose_proxy_logger.exception("Failed to send %s budget alert for %s", scope, entity_id)


class CustomerBudgetAlerting:
    @staticmethod
    def callbacks() -> tuple[CustomLogger, ...]:
        return tuple(
            callback
            for callback in litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)
            if type(callback).async_on_customer_budget_event is not CustomLogger.async_on_customer_budget_event
        )

    @staticmethod
    async def detect_committed_batch(
        transactions: Mapping[str, object],
        database: object,
        cache: CustomerBudgetClaims,
        *,
        now: datetime | None = None,
        windows: Sequence[CommittedWindowSpend] = (),
    ) -> None:
        if not CustomerBudgetAlerting.callbacks():
            return
        instant: Final = now if now is not None else utc_now()
        for level in _LEVELS:
            for entity_id, cost in _CHARGES.validate_python(transactions.get(level.transactions) or {}).items():
                if cost > 0:
                    await _detect_customer_events(level, entity_id, database, cache, instant)
        for scope in _APP_SCOPES:
            for entity_id, cost in _CHARGES.validate_python(
                transactions.get(f"{scope}_list_transactions") or {}
            ).items():
                if cost > 0:
                    await _detect_app_events(scope, entity_id, database, cache, instant)
        await CustomerBudgetAlerting.detect_committed_windows(windows, database, cache, now=instant)

    @staticmethod
    async def detect_committed_windows(
        transactions: Sequence[CommittedWindowSpend],
        database: object,
        cache: CustomerBudgetClaims,
        *,
        now: datetime | None = None,
    ) -> None:
        if not CustomerBudgetAlerting.callbacks():
            return
        instant: Final = now if now is not None else utc_now()
        for scope in _APP_SCOPES:
            for entity_id in frozenset(item.entity_id for item in transactions if item.entity_type == scope):
                await _detect_app_events(
                    scope,
                    entity_id,
                    database,
                    cache,
                    instant,
                    tuple(item for item in transactions if item.entity_type == scope and item.entity_id == entity_id),
                )

    @staticmethod
    @with_service_target("customer_budget_alerts")
    async def emit(event: CustomerBudgetEvent, cache: CustomerBudgetClaims) -> None:
        callbacks: Final = CustomerBudgetAlerting.callbacks()
        if not callbacks:
            return
        identity: Final = json.dumps(
            (event.event, event.scope, event.entity_id, event.period)
            + ((event.budget_duration,) if event.budget_duration is not None else ())
        )
        key: Final = "customer_budget_alerts:" + hashlib.sha256(identity.encode()).hexdigest()
        ttl: Final = (
            100 * 365 * 24 * 60 * 60
            if event.period == "lifetime"
            else max(1, math.ceil((datetime.fromisoformat(event.period) - event.timestamp).total_seconds()))
            + (SlackAlertingArgsEnum.budget_alert_ttl.value if event.scope in _APP_SCOPES else 0)
        )
        if await cache.async_increment_cache(key=key, value=1, ttl=ttl) != 1:
            return
        await asyncio.gather(*(CustomerBudgetAlerting._deliver(callback, event) for callback in callbacks))

    @staticmethod
    async def _deliver(callback: CustomLogger, event: CustomerBudgetEvent) -> None:
        try:
            await callback.async_on_customer_budget_event(event)
        except Exception:  # noqa: BLE001  # A failed callback must not replay committed spend or hide other callbacks.
            verbose_proxy_logger.exception("Failed to deliver %s customer budget event", event.event)
