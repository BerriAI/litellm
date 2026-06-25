from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from fastapi import HTTPException, status
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.proxy._types import (
    KeyRequestBase,
    LiteLLM_ManagementEndpoint_MetadataFields,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    NewProjectRequest,
    UpdateProjectRequest,
    UserAPIKeyAuth,
)
from litellm.proxy._types import (  # noqa: F401  re-exported
    user_api_key_has_admin_view as _user_has_admin_view,
)
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.utils import _premium_user_check
from litellm.repositories.team_repository import TeamRepository

if TYPE_CHECKING:
    from litellm.proxy._types import NewProjectRequest, UpdateProjectRequest
    from litellm.proxy.utils import PrismaClient, ProxyLogging


def require_caller_user_id_for_non_admin(
    user_api_key_dict: UserAPIKeyAuth,
) -> str:
    """Return the caller's user_id, or raise 403 if missing.

    Non-admin analytics endpoints scope queries by the caller's own user_id.
    Service-account keys are deliberately created with user_id=None
    (key_management_endpoints.py forces ``data.user_id = None`` at key
    creation). Without this guard, that None value flows through to the
    daily-activity builder, which treats ``entity_id is None`` as "no filter"
    and returns every tenant's data.

    Callers must check is_admin first; this helper is only valid on the
    non-admin scoping branch.
    """
    if user_api_key_dict.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    "Service-account keys cannot query user analytics. "
                    "Use a user-bound key, or call as a proxy admin."
                )
            },
        )
    return user_api_key_dict.user_id


def _check_passthrough_routes_caller_permission(
    data: BaseModel,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    entity: str = "key",
) -> None:
    """
    Only proxy admins may set `allowed_passthrough_routes` (top-level or under
    `metadata`) — it short-circuits the role-based route gate, so keys and teams
    must be gated identically.
    """
    # view-only admins excluded by design; blocked upstream from writes anyway
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return
    if getattr(data, "allowed_passthrough_routes", None):
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Only proxy admins can set `allowed_passthrough_routes` on a {entity}."
            },
        )
    metadata = getattr(data, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("allowed_passthrough_routes"):
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Only proxy admins can set `metadata.allowed_passthrough_routes` on a {entity}."
            },
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


async def _is_user_org_admin_for_team(
    user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable
) -> bool:
    """
    Check if user is an org admin for the team's organization.

    Returns True if:
    - The team belongs to an organization, AND
    - The user has org_admin role in that organization
    """
    if not team_obj.organization_id or not user_api_key_dict.user_id:
        return False

    from litellm.proxy.auth.auth_checks import get_user_object
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    caller_user = await get_user_object(
        user_id=user_api_key_dict.user_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        user_id_upsert=False,
        proxy_logging_obj=proxy_logging_obj,
    )
    if caller_user is None:
        return False

    for m in caller_user.organization_memberships or []:
        if (
            m.organization_id == team_obj.organization_id
            and m.user_role == LitellmUserRoles.ORG_ADMIN.value
        ):
            return True

    return False


def _team_member_has_permission(
    user_api_key_dict: UserAPIKeyAuth,
    team_obj: LiteLLM_TeamTable,
    permission: str,
) -> bool:
    """Check if a non-admin team member has a specific permission on a team."""
    if not team_obj.team_member_permissions:
        return False
    if permission not in team_obj.team_member_permissions:
        return False
    for member in team_obj.members_with_roles:
        if member.user_id is not None and member.user_id == user_api_key_dict.user_id:
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
            teams = await TeamRepository(prisma_client).table.find_many(
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

    teams = await TeamRepository(prisma_client).table.find_many(
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
        LiteLLM_ProjectTable,
        "NewProjectRequest",
        "UpdateProjectRequest",
    ],
    field_name: str,
    value: Any,
) -> None:
    """
    Helper function to set metadata fields that require premium user checks

    Args:
        object_data: The team/key/organization/project data object to modify
        field_name: Name of the metadata field to set
        value: Value to set for the field
    """
    if field_name in LiteLLM_ManagementEndpoint_MetadataFields_Premium and value:
        _premium_user_check(field_name)

    object_data.metadata = object_data.metadata or {}
    object_data.metadata[field_name] = value


_TEAM_MEMBER_BUDGET_LIMIT_FIELDS = (
    "max_budget",
    "soft_budget",
    "max_parallel_requests",
    "tpm_limit",
    "rpm_limit",
    "model_max_budget",
    "budget_duration",
    "allowed_models",
)


def _is_set_budget_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    return True


def _has_meaningful_budget_limit(budget_values: Dict[str, Any]) -> bool:
    """A budget is meaningful if at least one limit is actually set; an empty
    list (no model restriction) and None both count as unset."""
    return any(
        _is_set_budget_value(budget_values.get(field))
        for field in _TEAM_MEMBER_BUDGET_LIMIT_FIELDS
    )


async def _upsert_budget_and_membership(
    tx,
    *,
    team_id: str,
    user_id: str,
    existing_budget_id: Optional[str],
    user_api_key_dict: UserAPIKeyAuth,
    budget_patch: Dict[str, Any],
    team_default_budget_id: Optional[str] = None,
):
    """
    Apply a merge-patch of per-member budget fields to a team membership.

    ``budget_patch`` holds only the budget columns the caller explicitly sent
    (RFC 7396 semantics): a value sets the column, ``None`` clears it, and a
    column that is absent from the dict is left untouched. Once the patch is
    applied, if the budget has no meaningful limit left the member's private
    budget is disconnected so they fall back to the team default.

    ``team_default_budget_id`` is the team's shared default member budget id
    (from team metadata.team_member_budget_id). When the membership still
    points at it, we clone-on-write so editing one member's budget does not
    mutate the shared default that every other member points at.
    """
    if not budget_patch:
        return

    write_data = dict(budget_patch)
    if "budget_duration" in write_data:
        duration = write_data["budget_duration"]
        write_data["budget_reset_at"] = (
            get_budget_reset_time(budget_duration=duration)
            if duration is not None
            else None
        )

    is_shared_default = (
        existing_budget_id is not None
        and team_default_budget_id is not None
        and existing_budget_id == team_default_budget_id
    )

    async def _disconnect():
        await tx.litellm_teammembership.update(
            where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}},
            data={"litellm_budget_table": {"disconnect": True}},
        )

    if existing_budget_id is not None and not is_shared_default:
        existing_budget = await tx.litellm_budgettable.find_unique(
            where={"budget_id": existing_budget_id}
        )
        merged = existing_budget.model_dump() if existing_budget is not None else {}
        merged.update(write_data)
        if not _has_meaningful_budget_limit(merged):
            await _disconnect()
            return
        await tx.litellm_budgettable.update(
            where={"budget_id": existing_budget_id},
            data={"updated_by": user_api_key_dict.user_id or "", **write_data},
        )
        return

    create_data: Dict[str, Any] = {
        "created_by": user_api_key_dict.user_id or "",
        "updated_by": user_api_key_dict.user_id or "",
    }

    if is_shared_default:
        default_budget_row = await tx.litellm_budgettable.find_unique(
            where={"budget_id": existing_budget_id}
        )
        if default_budget_row is not None:
            default_budget_dict = default_budget_row.model_dump()
            for field in _TEAM_MEMBER_BUDGET_LIMIT_FIELDS:
                value = default_budget_dict.get(field)
                if _is_set_budget_value(value):
                    create_data[field] = value

    create_data.update(write_data)

    if create_data.get("budget_duration") is not None:
        create_data["budget_reset_at"] = get_budget_reset_time(
            budget_duration=create_data["budget_duration"]
        )
    else:
        create_data.pop("budget_reset_at", None)

    if not _has_meaningful_budget_limit(create_data):
        if existing_budget_id is not None:
            await _disconnect()
        return

    new_budget = await tx.litellm_budgettable.create(
        data=create_data,
        include={"team_membership": True},
    )
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
        # The UI sends falsy defaults (False, [], {}) even when the user has not
        # enabled any enterprise feature (see #20304, #30285); require a license
        # only for a truthy value. The falsy value is still persisted below so a
        # previously-set field can be cleared.
        if updated_kv.get(field_name):
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
