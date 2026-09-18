"""Pins the budget state auth resolved onto ``UserAPIKeyAuth`` and emits it as request metadata."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.team import BudgetLimitEntry, LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.proxy.carried_budget_state import (
    OrgBudgetSnapshot,
    TeamBudgetSnapshot,
    UserBudgetSnapshot,
)

USER_BUDGET_LIMITS_METADATA_KEY: Final = "user_api_key_user_budget_limits"
_BUDGET_LIMITS_ADAPTER: Final = TypeAdapter(list[BudgetLimitEntry])


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


def carry_organization_budget_state(valid_token: UserAPIKeyAuth, org_table: LiteLLM_OrganizationTable) -> None:
    budget_table: Final = org_table.litellm_budget_table
    valid_token.organization_alias = (
        org_table.organization_alias
    )  # rebind-ok: the request credential is pinned in place
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


def carried_user_budget_limits_metadata(valid_token: UserAPIKeyAuth) -> tuple[dict[str, object], ...] | None:
    if valid_token.user_budget_limits is None:
        return None
    try:
        windows: Final = _BUDGET_LIMITS_ADAPTER.validate_python(valid_token.user_budget_limits)
    except ValidationError:
        return None
    return tuple(window.model_dump(mode="json") for window in windows)


def carried_user_budget_limits(metadata: Mapping[str, object]) -> tuple[BudgetLimitEntry, ...] | None:
    raw: Final = metadata.get(USER_BUDGET_LIMITS_METADATA_KEY)
    if raw is None:
        return None
    try:
        return tuple(_BUDGET_LIMITS_ADAPTER.validate_python(raw))
    except ValidationError:
        return None
