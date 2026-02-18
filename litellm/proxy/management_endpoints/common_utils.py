from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.proxy._types import (
    KeyRequestBase,
    LiteLLM_ManagementEndpoint_MetadataFields,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.utils import _premium_user_check

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging


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


async def _user_has_admin_privileges(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Optional["PrismaClient"] = None,
    user_api_key_cache: Optional["DualCache"] = None,
    proxy_logging_obj: Optional["ProxyLogging"] = None,
) -> bool:
    """
    Check if user has admin privileges (proxy admin, team admin, or org admin).

    Args:
        user_api_key_dict: User API key authentication object
        prisma_client: Prisma client for database operations
        user_api_key_cache: Cache for user API keys
        proxy_logging_obj: Proxy logging object

    Returns:
        True if user is proxy admin, team admin for any team, or org admin for any organization
    """
    # Check if user is proxy admin
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        return True

    # If no database connection, can't check team/org admin status
    if prisma_client is None or user_api_key_dict.user_id is None:
        return False

    # Get user object to check team and org admin status
    from litellm.caching import DualCache as DualCacheImport
    from litellm.proxy.auth.auth_checks import get_user_object

    try:
        user_obj = await get_user_object(
            user_id=user_api_key_dict.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache or DualCacheImport(),
            user_id_upsert=False,
            proxy_logging_obj=proxy_logging_obj,
        )

        if user_obj is None:
            return False

        # Check if user is org admin for any organization
        if user_obj.organization_memberships is not None:
            for membership in user_obj.organization_memberships:
                if membership.user_role == LitellmUserRoles.ORG_ADMIN.value:
                    return True

        # Check if user is team admin for any team
        if user_obj.teams is not None and len(user_obj.teams) > 0:
            # Get all teams user is in
            teams = await prisma_client.db.litellm_teamtable.find_many(
                where={"team_id": {"in": user_obj.teams}}
            )

            for team in teams:
                team_obj = LiteLLM_TeamTable(**team.model_dump())
                if _is_user_team_admin(
                    user_api_key_dict=user_api_key_dict, team_obj=team_obj
                ):
                    return True

    except Exception as e:
        # If there's an error checking, default to False for security
        verbose_proxy_logger.debug(
            f"Error checking admin privileges for user {user_api_key_dict.user_id}: {e}"
        )
        return False

    return False


def _org_admin_can_invite_user(
    admin_user_obj: LiteLLM_UserTable,
    target_user_obj: LiteLLM_UserTable,
) -> bool:
    """
    Check if an org admin can invite the target user.
    Target user must be in at least one org where the admin has org admin role.

    Args:
        admin_user_obj: The admin user's full object (from get_user_object)
        target_user_obj: The target user's full object (from get_user_object)

    Returns:
        True if target user is in an org where admin has org admin role
    """
    if admin_user_obj.organization_memberships is None:
        return False
    admin_org_ids = {
        m.organization_id
        for m in admin_user_obj.organization_memberships
        if m.user_role == LitellmUserRoles.ORG_ADMIN.value
    }
    if not admin_org_ids:
        return False
    if target_user_obj.organization_memberships is None:
        return False
    target_org_ids = {
        m.organization_id for m in target_user_obj.organization_memberships
    }
    return bool(admin_org_ids & target_org_ids)


async def _team_admin_can_invite_user(
    user_api_key_dict: UserAPIKeyAuth,
    admin_user_obj: LiteLLM_UserTable,
    target_user_obj: LiteLLM_UserTable,
    prisma_client: "PrismaClient",
) -> bool:
    """
    Check if a team admin can invite the target user.
    Target user must be in at least one team where the admin has team admin role.

    Args:
        user_api_key_dict: The admin user's API key auth object
        admin_user_obj: The admin user's full object (from get_user_object)
        target_user_obj: The target user's full object (from get_user_object)
        prisma_client: Prisma client for database operations

    Returns:
        True if target user is in a team where admin has team admin role
    """
    if not admin_user_obj.teams or len(admin_user_obj.teams) == 0:
        return False
    if not target_user_obj.teams or len(target_user_obj.teams) == 0:
        return False

    teams = await prisma_client.db.litellm_teamtable.find_many(
        where={"team_id": {"in": admin_user_obj.teams}}
    )
    admin_team_ids = [
        team.team_id
        for team in teams
        if _is_user_team_admin(
            user_api_key_dict=user_api_key_dict,
            team_obj=LiteLLM_TeamTable(**team.model_dump()),
        )
    ]
    if not admin_team_ids:
        return False
    target_team_ids = set(target_user_obj.teams)
    return bool(set(admin_team_ids) & target_team_ids)


async def admin_can_invite_user(
    target_user_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Optional["PrismaClient"] = None,
    user_api_key_cache: Optional["DualCache"] = None,
    proxy_logging_obj: Optional["ProxyLogging"] = None,
) -> bool:
    """
    Check if the admin can create an invitation for the target user.
    - Proxy admins: can invite any user
    - Org admins: can only invite users in their org(s)
    - Team admins: can only invite users in their team(s)

    Uses get_user_object for caching of both admin and target user objects.

    Args:
        target_user_id: The user_id of the user to invite
        user_api_key_dict: The admin user's API key auth object
        prisma_client: Prisma client for database operations
        user_api_key_cache: Cache for user API keys
        proxy_logging_obj: Proxy logging object

    Returns:
        True if user can invite the target user
    """
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        return True

    if prisma_client is None or user_api_key_dict.user_id is None:
        return False

    from litellm.caching import DualCache as DualCacheImport
    from litellm.proxy.auth.auth_checks import get_user_object

    try:
        cache = user_api_key_cache or DualCacheImport()
        admin_user_obj = await get_user_object(
            user_id=user_api_key_dict.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=cache,
            user_id_upsert=False,
            proxy_logging_obj=proxy_logging_obj,
        )
        if admin_user_obj is None:
            return False

        target_user_obj = await get_user_object(
            user_id=target_user_id,
            prisma_client=prisma_client,
            user_api_key_cache=cache,
            user_id_upsert=False,
            proxy_logging_obj=proxy_logging_obj,
        )
        if target_user_obj is None:
            return False

        if _org_admin_can_invite_user(admin_user_obj, target_user_obj):
            return True

        if await _team_admin_can_invite_user(
            user_api_key_dict=user_api_key_dict,
            admin_user_obj=admin_user_obj,
            target_user_obj=target_user_obj,
            prisma_client=prisma_client,
        ):
            return True

        return False
    except Exception as e:
        verbose_proxy_logger.debug(
            f"Error checking invite permission for user {user_api_key_dict.user_id}: {e}"
        )
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
        value = updated_kv.get(field_name)
        # Skip the premium check for empty collections ([] or {}).
        # The UI sends these as defaults even when the user hasn't configured
        # any enterprise features (see issue #20304).  However, we still
        # proceed with the update so that users can intentionally clear a
        # previously-set field by sending an empty list/dict.
        if value is not None and value != [] and value != {}:
            _premium_user_check()

    if field_name in updated_kv and updated_kv[field_name] is not None:
        # remove field from updated_kv
        _value = updated_kv.pop(field_name)
        if "metadata" in updated_kv and updated_kv["metadata"] is not None:
            updated_kv["metadata"][field_name] = _value
        else:
            updated_kv["metadata"] = {field_name: _value}


def _has_non_empty_value(value: Any) -> bool:
    """Check if a value has real content (not None, not empty list, not blank string)."""
    if value is None:
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


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
