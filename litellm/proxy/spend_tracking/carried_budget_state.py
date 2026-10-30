"""Pins the budget state auth resolved onto ``UserAPIKeyAuth`` and emits it as request metadata."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.proxy.carried_budget_state import (
    LiveBudgetSpend,
    OrgBudgetSnapshot,
    TeamBudgetSnapshot,
    UserBudgetSnapshot,
)


def carry_team_and_user_budget_state(
    valid_token: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
) -> None:
    if team_object is not None:
        valid_token.team_budget_snapshot = TeamBudgetSnapshot(  # rebind-ok: the request credential is pinned in place
            budget_reset_at=team_object.budget_reset_at,
            max_budget=team_object.max_budget,
        )
        valid_token.team_model_max_budget = team_object.model_max_budget  # rebind-ok: caller keeps this object
    if user_object is not None:
        valid_token.user_budget_snapshot = UserBudgetSnapshot(  # rebind-ok: same object the caller keeps using
            budget_reset_at=user_object.budget_reset_at,
            max_budget=user_object.max_budget,
            user_alias=user_object.user_alias,
        )


@dataclass(frozen=True, slots=True)
class _BudgetedSpend:
    counter_key: str
    stored_spend: float


async def _read_live_spend(
    budgeted: _BudgetedSpend | None, read_spend: Callable[[str, float], Awaitable[float]]
) -> float | None:
    if budgeted is None:
        return None
    return max(await read_spend(budgeted.counter_key, budgeted.stored_spend), budgeted.stored_spend)


def _org_budgeted_spend(valid_token: UserAPIKeyAuth, team_object: LiteLLM_TeamTable | None) -> _BudgetedSpend | None:
    org_snapshot: Final = valid_token.org_budget_snapshot
    org_id: Final = valid_token.org_id or (team_object.organization_id if team_object is not None else None)
    if org_snapshot is None or org_snapshot.max_budget is None or org_id is None:
        return None
    return _BudgetedSpend(f"spend:org:{org_id}", org_snapshot.spend)


def _end_user_budgeted_spend(end_user_object: LiteLLM_EndUserTable | None) -> _BudgetedSpend | None:
    budget_table: Final = end_user_object.litellm_budget_table if end_user_object is not None else None
    if end_user_object is None or budget_table is None or budget_table.max_budget is None:
        return None
    return _BudgetedSpend(f"spend:end_user:{end_user_object.user_id}", end_user_object.spend)


async def carry_live_budget_spend(
    valid_token: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
    end_user_object: LiteLLM_EndUserTable | None,
    read_spend: Callable[[str, float], Awaitable[float]],
) -> None:
    budgeted: Final = (
        _BudgetedSpend(f"spend:key:{valid_token.token}", valid_token.spend or 0.0)
        if valid_token.max_budget is not None and valid_token.token is not None
        else None,
        _BudgetedSpend(f"spend:team:{team_object.team_id}", team_object.spend or 0.0)
        if team_object is not None and team_object.max_budget is not None
        else None,
        _BudgetedSpend(f"spend:user:{user_object.user_id}", user_object.spend or 0.0)
        if user_object is not None and user_object.max_budget is not None
        else None,
        _org_budgeted_spend(valid_token=valid_token, team_object=team_object),
        _end_user_budgeted_spend(end_user_object=end_user_object),
    )
    key, team, user, org, end_user = await asyncio.gather(
        *(_read_live_spend(budgeted=entity, read_spend=read_spend) for entity in budgeted)
    )
    valid_token.live_budget_spend = LiveBudgetSpend(  # rebind-ok: the request credential is pinned in place
        key=key, team=team, user=user, org=org, end_user=end_user
    )


def carry_organization_budget_state(valid_token: UserAPIKeyAuth, org_table: LiteLLM_OrganizationTable) -> None:
    budget_table: Final = org_table.litellm_budget_table
    valid_token.organization_alias = org_table.organization_alias
    valid_token.org_budget_snapshot = OrgBudgetSnapshot(  # rebind-ok: same object the caller keeps using
        spend=org_table.spend,
        max_budget=budget_table.max_budget if budget_table is not None else None,
    )


def carried_budget_metadata(valid_token: UserAPIKeyAuth) -> Mapping[str, object]:
    snapshots: Final = (
        valid_token.team_budget_snapshot,
        valid_token.user_budget_snapshot,
        valid_token.org_budget_snapshot,
        valid_token.live_budget_spend,
    )
    return MappingProxyType(
        {
            key: value
            for snapshot in snapshots
            if snapshot is not None
            for key, value in snapshot.metadata_entries().items()
        }
    )
