from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import chain
from typing import TYPE_CHECKING, Final, TypeVar

from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.auth.customer_hierarchy import CustomerScopeKey
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    CustomerBudgetUsage,
    CustomerMember,
    Organization,
    Team,
)

if TYPE_CHECKING:
    from prisma.models import LiteLLM_BudgetWindowSpend

    from litellm.proxy.spend_tracking.budget_reservation import BudgetCounter

Entity: Final = TypeVar("Entity", Organization, Team, CustomerMember)


def _identity(entity: Organization | Team | CustomerMember) -> tuple[CustomerScopeKey, str, Budget | None]:
    if isinstance(entity, Organization):
        return "customer_org", entity.customer_org_id, entity.budget
    if isinstance(entity, Team):
        return "customer_team", entity.customer_team_id, entity.effective_budget
    return "customer_member", entity.customer_member_id, entity.effective_budget


def _scalar_usage(
    entity: Organization | Team | CustomerMember, budget: Budget | None
) -> tuple[CustomerBudgetUsage, ...]:
    if budget is None or budget.max_budget is None:
        return ()
    return (
        CustomerBudgetUsage(
            source="scalar",
            budget_duration=budget.budget_duration,
            max_budget=budget.max_budget,
            spend=entity.spend,
            reset_at=entity.budget_reset_at,
        ),
    )


def _usage(
    counters: Sequence["BudgetCounter"],
    rows: Mapping[tuple[str, str, str], "LiteLLM_BudgetWindowSpend"],
) -> tuple[CustomerBudgetUsage, ...]:
    from litellm.proxy.db.spend_counter_reseed import recorded_window_spend

    return tuple(
        CustomerBudgetUsage(
            source="window",
            budget_duration=counter.window_duration,
            max_budget=counter.max_budget,
            spend=recorded_window_spend(
                rows.get((counter.entity_type, counter.entity_id, counter.window_duration or "")),
                counter.entity_type,
                counter.window_start,
            ),
            window_start=counter.window_start,
            reset_at=counter.reset_at,
        )
        for counter in counters
    )


async def customer_budget_usage(
    entities: Sequence[Entity],
    table: TableActions["LiteLLM_BudgetWindowSpend"],
    now: datetime,
    settings: BudgetResetSettings,
) -> tuple[Entity, ...]:
    from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters

    if not entities:
        return ()
    identities: Final = tuple(_identity(entity) for entity in entities)
    counters: Final = tuple(
        customer_window_counters(scope, entity_id, budget.budget_limits or (), now=now, reset_settings=settings)
        if budget is not None
        else ()
        for scope, entity_id, budget in identities
    )
    windows: Final = tuple(chain.from_iterable(counters))
    rows: Final = await find_many_in(
        table,
        "entity_id",
        tuple(counter.entity_id for counter in windows),
        where={"entity_type": identities[0][0]},
    )
    by_window: Final = {(row.entity_type, row.entity_id, row.window_duration): row for row in rows}
    return tuple(
        entity.model_copy(
            update={"budget_usage": _scalar_usage(entity, identity[2]) + _usage(entity_windows, by_window)}
        )
        for entity, identity, entity_windows in zip(entities, identities, counters, strict=True)
    )
