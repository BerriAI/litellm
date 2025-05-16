from typing import Any, Union, Optional

from litellm.proxy._types import (
    GenerateKeyRequest,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.utils import _premium_user_check


def _user_has_admin_view(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    )


def _is_user_team_admin(
    user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable
) -> bool:
    for member in team_obj.members_with_roles:
        if (
            member.user_id is not None and member.user_id == user_api_key_dict.user_id
        ) and member.role == "admin":
            return True

    return False


def _set_object_metadata_field(
    object_data: Union[LiteLLM_TeamTable, GenerateKeyRequest],
    field_name: str,
    value: Any,
) -> None:
    """
    Helper function to set metadata fields that require premium user checks

    Args:
        object_data: The team data object to modify
        field_name: Name of the metadata field to set
        value: Value to set for the field
    """
    if field_name in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
        _premium_user_check()
    object_data.metadata = object_data.metadata or {}
    object_data.metadata[field_name] = value


async def _upsert_budget_and_membership(
    tx,
    *,
    team_id: str,
    user_id: str,
    max_budget: Optional[float],
    existing_budget_id: Optional[str],
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Helper function to Create/Update or Delete the budget within the team membership
    Args:
        tx: The transaction object
        team_id: The ID of the team
        user_id: The ID of the user
        max_budget: The maximum budget for the team
        existing_budget_id: The ID of the existing budget, if any
        user_api_key_dict: User API Key dictionary containing user information

    If max_budget is None, the user's budget is removed from the team membership.
    If max_budget exists, a budget is updated or created and linked to the team membership.
    """
    if max_budget is None:
        # disconnect the budget since max_budget is None
        await tx.litellm_teammembership.update(
            where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}},
            data={"litellm_budget_table": {"disconnect": True}},
        )
        return

    if existing_budget_id:
        # update the existing budget
        await tx.litellm_budgettable.update(
            where={"budget_id": existing_budget_id},
            data={"max_budget": max_budget},
        )
        return

    # create a new budget
    new_budget = await tx.litellm_budgettable.create(
        data={
            "max_budget": max_budget,
            "created_by": user_api_key_dict.user_id or "",
            "updated_by": user_api_key_dict.user_id or "",
        },
        include={"team_membership": True},
    )
    # upsert the team membership with the new/updated budget
    await tx.litellm_teammembership.upsert(
        where={
            "user_id_team_id": {
                "user_id": user_id,
                "team_id": team_id,
            }
        },
        data={
            "create": {
                "user_id": user_id,
                "team_id": team_id,
                "litellm_budget_table": {
                    "connect": {"budget_id": new_budget.budget_id},
                },
            },
            "update": {
                "litellm_budget_table": {
                    "connect": {"budget_id": new_budget.budget_id},
                },
            },
        },
    )
