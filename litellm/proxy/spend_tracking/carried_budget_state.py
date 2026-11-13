"""Pins the budget state auth resolved onto ``UserAPIKeyAuth`` and emits it as request metadata."""

from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # legacy shared helper; its output is validated below
)
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    customer_hierarchy_ids_from_request,
)
from litellm.types.proxy.carried_budget_state import (
    PERSON_BUDGET_WINDOWS_METADATA_KEY,
    CommittedBudgetSpend,
    EntitySpendCounterKeys,
    OrgBudgetSnapshot,
    PersonBudgetWindow,
    TeamBudgetSnapshot,
    UserBudgetSnapshot,
)

_PERSON_WINDOWS: Final = TypeAdapter(tuple[PersonBudgetWindow, ...])
_METADATA: Final = TypeAdapter(Mapping[str, object])
_CUSTOMER_FIELDS: Final = (
    "user_api_key_customer_org_id",
    "user_api_key_customer_team_id",
    "user_api_key_customer_member_id",
)


def person_budget_windows_from_kwargs(kwargs: Mapping[str, object]) -> tuple[PersonBudgetWindow, ...]:
    """Read the authenticated snapshot, including serialized spend-event replay."""
    metadata: Final = _METADATA.validate_python(get_litellm_metadata_from_kwargs(dict(kwargs)) or {})
    standard: Final = as_str_mapping(kwargs.get("standard_logging_object")) or {}
    fallback: Final = as_str_mapping(standard.get("metadata")) or {}
    windows: Final = metadata.get(PERSON_BUDGET_WINDOWS_METADATA_KEY, fallback.get(PERSON_BUDGET_WINDOWS_METADATA_KEY))
    return _PERSON_WINDOWS.validate_python(windows) if windows else ()


def creator_budget_metadata(valid_token: UserAPIKeyAuth) -> Mapping[str, object]:
    """Only authenticated creator state is eligible for durable deferred attribution."""
    customer: Final = valid_token.customer_hierarchy
    customer_ids: Final = (
        {
            "user_api_key_customer_org_id": customer.customer_org_id,
            "user_api_key_customer_team_id": customer.customer_team_id,
            "user_api_key_customer_member_id": customer.customer_member_id,
        }
        if customer is not None
        else {}
    )
    return {
        **(
            {
                PERSON_BUDGET_WINDOWS_METADATA_KEY: [
                    window.model_dump(mode="json") for window in valid_token.person_budget_windows
                ]
            }
            if valid_token.person_budget_windows
            else {}
        ),
        **customer_ids,
    }


def stored_budget_metadata(value: object) -> Mapping[str, object]:
    """Validate the saved personal windows and customer IDs; old managed rows have neither."""
    if value is None:
        return {}
    metadata: Final = _METADATA.validate_python(value or {})
    windows: Final = _PERSON_WINDOWS.validate_python(metadata.get(PERSON_BUDGET_WINDOWS_METADATA_KEY) or ())
    return {
        **(
            {PERSON_BUDGET_WINDOWS_METADATA_KEY: [window.model_dump(mode="json") for window in windows]}
            if windows
            else {}
        ),
        **_customer_metadata(metadata),
    }


def _customer_metadata(metadata: Mapping[str, object]) -> Mapping[str, str]:
    return {field: value for field in _CUSTOMER_FIELDS if isinstance(value := metadata.get(field), str) and value}


def restore_creator_budget_metadata(valid_token: UserAPIKeyAuth, metadata: Mapping[str, object]) -> None:
    """Restore trusted internal metadata after auth construction strips request-only fields."""
    valid_token.person_budget_windows = _PERSON_WINDOWS.validate_python(
        metadata.get(PERSON_BUDGET_WINDOWS_METADATA_KEY) or ()
    )
    customer: Final = customer_hierarchy_ids_from_request({"litellm_params": {"metadata": metadata}})
    valid_token.customer_hierarchy = (
        CustomerHierarchyAuth(
            customer_org_id=customer.customer_org_id,
            customer_team_id=customer.customer_team_id,
            customer_member_id=customer.customer_member_id,
            scopes=(),
        )
        if customer is not None
        else None
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
    """Overwrite request metadata from auth, including clearing an unassigned person's windows."""
    snapshots: Final = (
        valid_token.team_budget_snapshot,
        valid_token.user_budget_snapshot,
        valid_token.org_budget_snapshot,
    )
    return MappingProxyType(
        {
            PERSON_BUDGET_WINDOWS_METADATA_KEY: [
                window.model_dump(mode="json") for window in valid_token.person_budget_windows
            ],
            **{
                key: value
                for snapshot in snapshots
                if snapshot is not None
                for key, value in snapshot.metadata_entries().items()
            },
        }
    )
