"""Pins the budget state auth resolved onto ``UserAPIKeyAuth`` and emits it as request metadata."""

from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.proxy.carried_budget_state import (
    CommittedBudgetSpend,
    EntitySpendCounterKeys,
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


def _budgeted_org_id(valid_token: UserAPIKeyAuth, team_object: LiteLLM_TeamTable | None) -> str | None:
    org_snapshot: Final = valid_token.org_budget_snapshot
    if org_snapshot is None or org_snapshot.max_budget is None:
        return None
    return valid_token.org_id or (team_object.organization_id if team_object is not None else None)


def _budgeted_end_user_id(end_user_object: LiteLLM_EndUserTable | None) -> str | None:
    budget_table: Final = end_user_object.litellm_budget_table if end_user_object is not None else None
    if end_user_object is None or budget_table is None or budget_table.max_budget is None:
        return None
    return end_user_object.user_id


async def carry_committed_budget_spend(
    valid_token: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
    end_user_object: LiteLLM_EndUserTable | None,
    read_committed: Callable[[EntitySpendCounterKeys], Awaitable[CommittedBudgetSpend]] | None,
) -> None:
    counter_keys: Final = EntitySpendCounterKeys.of(
        token=valid_token.token if valid_token.max_budget is not None else None,
        team_id=team_object.team_id if team_object is not None and team_object.max_budget is not None else None,
        user_id=user_object.user_id if user_object is not None and user_object.max_budget is not None else None,
        org_id=_budgeted_org_id(valid_token=valid_token, team_object=team_object),
        end_user_id=_budgeted_end_user_id(end_user_object=end_user_object),
    )
    valid_token.committed_budget_spend = (  # rebind-ok: request-scoped credential
        CommittedBudgetSpend(counter_keys=counter_keys)
        if read_committed is None
        else await read_committed(counter_keys)
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
