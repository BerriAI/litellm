from typing import Any, Dict, Optional, Union

from litellm.proxy._types import (
    KeyRequestBase,
    LiteLLM_ManagementEndpoint_MetadataFields,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    LiteLLM_OrganizationTable,
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
    object_data: Union[
        LiteLLM_TeamTable,
        KeyRequestBase,
        LiteLLM_OrganizationTable,
    ],
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
        _premium_user_check(field_name)

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
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
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
        tpm_limit: Tokens per minute limit for the team member
        rpm_limit: Requests per minute limit for the team member

    If max_budget, tpm_limit, and rpm_limit are all None, the user's budget is removed from the team membership.
    If any of these values exist, a budget is updated or created and linked to the team membership.
    """
    if max_budget is None and tpm_limit is None and rpm_limit is None:
        # disconnect the budget since all limits are None
        await tx.litellm_teammembership.update(
            where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}},
            data={"litellm_budget_table": {"disconnect": True}},
        )
        return

    # create a new budget
    create_data: Dict[str, Any] = {
        "created_by": user_api_key_dict.user_id or "",
        "updated_by": user_api_key_dict.user_id or "",
    }
    if max_budget is not None:
        create_data["max_budget"] = max_budget
    if tpm_limit is not None:
        create_data["tpm_limit"] = tpm_limit
    if rpm_limit is not None:
        create_data["rpm_limit"] = rpm_limit

    new_budget = await tx.litellm_budgettable.create(
        data=create_data,
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


def _update_metadata_field(updated_kv: dict, field_name: str) -> None:
    """
    Helper function to update metadata fields that require premium user checks in the update endpoint

    Args:
        updated_kv: The key-value dict being used for the update
        field_name: Name of the metadata field being updated
    """
    if field_name in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
        _premium_user_check()

    if field_name in updated_kv and updated_kv[field_name] is not None:
        # remove field from updated_kv
        _value = updated_kv.pop(field_name)
        if "metadata" in updated_kv and updated_kv["metadata"] is not None:
            updated_kv["metadata"][field_name] = _value
        else:
            updated_kv["metadata"] = {field_name: _value}


def _update_metadata_fields(updated_kv: dict) -> None:
    """
    Helper function to update all metadata fields (both premium and standard).

    Args:
        updated_kv: The key-value dict being used for the update
    """
    for field in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
        if field in updated_kv and updated_kv[field] is not None:
            _update_metadata_field(updated_kv=updated_kv, field_name=field)

    for field in LiteLLM_ManagementEndpoint_MetadataFields:
        if field in updated_kv and updated_kv[field] is not None:
            _update_metadata_field(updated_kv=updated_kv, field_name=field)
