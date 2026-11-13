from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, InstanceOf, TypeAdapter

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.proxy.management_helpers.customer_hierarchy import (
    PrismaRecord,
    effective_member_budget,
    related_budget,
    utc_now,
)
from litellm.types.integrations.slack_alerting import AlertType, CallInfo, Litellm_EntityType


class CustomerBudgetAlerts(Protocol):
    @property
    def alerting(self) -> Sequence[str] | None: ...

    @property
    def alert_types(self) -> Sequence[AlertType]: ...

    async def budget_alerts(self, type: Literal["soft_budget", "token_budget"], user_info: CallInfo) -> object: ...


class CustomerBudgetClaims(Protocol):
    async def async_increment_cache(self, *, key: str, value: float, ttl: float) -> float | None: ...


@with_service_target("customer_budget_alerts")
async def claim_customer_budget_alert(cache: CustomerBudgetClaims, key: str, ttl: float) -> bool:
    return await cache.async_increment_cache(key=key, value=1, ttl=ttl) == 1


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
    budget_reset_at: datetime | None = None
    budget: Mapping[str, object] | None = None
    budget_override: Mapping[str, object] | None = None
    budget_override_expires_at: datetime | None = None
    parent_org: "_CustomerRow | None" = None
    parent_team: "_CustomerRow | None" = None


@dataclass(frozen=True, slots=True)
class _Level:
    event_group: Litellm_EntityType
    transactions: str
    id_column: str
    table: str
    include: Mapping[str, object]


_LEVELS: Final = (
    _Level(
        Litellm_EntityType.CUSTOMER_ORG,
        "customer_org_list_transactions",
        "customer_org_id",
        "litellm_customerorganizationtable",
        {"budget": True},
    ),
    _Level(
        Litellm_EntityType.CUSTOMER_TEAM,
        "customer_team_list_transactions",
        "customer_team_id",
        "litellm_customerteamtable",
        {"budget": True, "parent_org": True},
    ),
    _Level(
        Litellm_EntityType.CUSTOMER_MEMBER,
        "customer_member_list_transactions",
        "customer_member_id",
        "litellm_customermembertable",
        {"budget": True, "budget_override": True, "parent_team": {"include": {"parent_org": True}}},
    ),
)
_CUSTOMER_GROUPS: Final = frozenset(level.event_group for level in _LEVELS)
_CHARGES: Final = TypeAdapter(Mapping[str, float])
_TABLE: Final = TypeAdapter(InstanceOf[_CustomerBudgetTable])


def customer_budget_alert_id(info: CallInfo) -> str | None:
    if info.event_group not in _CUSTOMER_GROUPS or info.entity_id is None:
        return None
    period: Final = info.customer_budget_reset_at.isoformat() if info.customer_budget_reset_at else "lifetime"
    return f"{info.event_group.value}:{info.entity_id}:{period}"


def customer_budget_event_info(info: CallInfo, *, soft: bool) -> CallInfo:
    if customer_budget_alert_id(info) is None:
        return info
    return info.model_copy(update={"max_budget": None} if soft else {"soft_budget": None})


def _call_info(level: _Level, entity_id: str, row: _CustomerRow, now: datetime) -> CallInfo:
    template: Final = related_budget(row.budget, now)
    override: Final = related_budget(row.budget_override, now, template.max_budget if template is not None else None)
    budget: Final = (
        effective_member_budget(template, override, row.budget_override_expires_at, now)
        if level.event_group == Litellm_EntityType.CUSTOMER_MEMBER
        else template
    )
    customer_team: Final = row.parent_team if level.event_group == Litellm_EntityType.CUSTOMER_MEMBER else row
    customer_org: Final = (
        row
        if level.event_group == Litellm_EntityType.CUSTOMER_ORG
        else customer_team.parent_org
        if customer_team
        else None
    )
    return CallInfo(
        event_group=level.event_group,
        entity_id=entity_id,
        customer_org_id=customer_org.customer_org_id if customer_org else None,
        customer_org_alias=customer_org.alias if customer_org else None,
        customer_team_id=(
            customer_team.customer_team_id
            if level.event_group != Litellm_EntityType.CUSTOMER_ORG and customer_team
            else None
        ),
        customer_member_id=entity_id if level.event_group == Litellm_EntityType.CUSTOMER_MEMBER else None,
        customer_budget_reset_at=row.budget_reset_at,
        spend=row.spend,
        max_budget=budget.max_budget if budget else None,
        soft_budget=budget.soft_budget if budget else None,
    )


async def _send_customer_alert(
    level: _Level, entity_id: str, database: object, alerts: CustomerBudgetAlerts, now: datetime
) -> None:
    try:
        table: Final = _TABLE.validate_python(getattr(database, level.table))
        record: Final = await table.find_unique(where={level.id_column: entity_id}, include=level.include)
        if record is None:
            return
        info: Final = _call_info(level, entity_id, _CustomerRow.model_validate(record.model_dump()), now)
        if info.soft_budget is not None:
            await alerts.budget_alerts(type="soft_budget", user_info=info)
        if info.max_budget is not None:
            await alerts.budget_alerts(type="token_budget", user_info=info)
    except Exception:  # noqa: BLE001  # Notification failures must not replay committed charges.
        verbose_proxy_logger.exception("Failed to send %s budget alert for %s", level.event_group.value, entity_id)


async def send_customer_budget_alerts(
    transactions: Mapping[str, object], database: object, alerts: CustomerBudgetAlerts, *, now: datetime | None = None
) -> None:
    if not alerts.alerting or AlertType.budget_alerts not in alerts.alert_types:
        return
    instant: Final = now if now is not None else utc_now()
    for level in _LEVELS:
        for entity_id, cost in _CHARGES.validate_python(transactions.get(level.transactions) or {}).items():
            if cost > 0:
                await _send_customer_alert(level, entity_id, database, alerts, instant)
