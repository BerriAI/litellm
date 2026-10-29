"""Pins the budget state auth resolved onto ``UserAPIKeyAuth`` and emits it as request metadata."""

from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.proxy.carried_budget_state import (
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


async def carry_live_key_and_team_spend(
    valid_token: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    read_spend: Callable[[str, float], Awaitable[float]],
) -> None:
    if valid_token.max_budget is not None and valid_token.token is not None:
        valid_token.spend = await read_spend(  # rebind-ok: the budget gauges and spend header read this token
            f"spend:key:{valid_token.token}", valid_token.spend or 0.0
        )
    if team_object is not None and team_object.max_budget is not None:
        valid_token.team_spend = await read_spend(  # rebind-ok: the team budget gauge reads this token
            f"spend:team:{team_object.team_id}", team_object.spend or 0.0
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
    )
    return MappingProxyType(
        {
            key: value
            for snapshot in snapshots
            if snapshot is not None
            for key, value in snapshot.metadata_entries().items()
        }
    )
