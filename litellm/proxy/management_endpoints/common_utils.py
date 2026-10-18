import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Optional, Union

from fastapi import HTTPException, status
from pydantic import BaseModel, TypeAdapter, ValidationError


# Defined above the `litellm.proxy.*` imports so the name is bound even when
# this module is imported first through the proxy import cycle (CodeQL:
# module-level cyclic import). Depends only on `math` + `HTTPException`.
def validate_finite_spend(spend: float | None) -> None:
    """Reject NaN/±inf spend before it reaches the DB / spend counter.

    A non-finite spend would otherwise slip past `spend >= max_budget`
    enforcement, since any comparison with NaN (and `-inf >= max_budget`)
    is False, letting the entity keep spending past its configured budget.
    """
    if spend is not None and not math.isfinite(spend):
        raise HTTPException(
            status_code=400,
            detail={"error": f"spend must be a finite number. Received: {spend}"},
        )


def validate_budget_duration(budget_duration: str | None, status_code: int = 400) -> None:
    """Reject budget durations that can't be parsed, are non-positive, or
    overflow date math, so a bad value can't be persisted and later crash the
    budget reset job.

    A non-positive duration also resolves to a reset time of "now", which leaves
    the row permanently due: the reset job re-reads it every tick and, once
    enough of them exist, they fill each batch and starve every other tenant's
    reset.
    """
    from litellm.proxy.common_utils.timezone_utils import budget_duration_error

    error: Final = budget_duration_error(budget_duration)
    if error is not None:
        raise HTTPException(status_code=status_code, detail={"error": error})


from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.proxy._types import (  # re-exported
    CommonProxyErrors,
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
    user_api_key_has_admin_view,
)
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.management.teams.authz import is_team_admin
from litellm.proxy.utils import (  # noqa: F401  # legacy module exports
    _premium_user_check,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    premium_user_check,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.types.utils import BudgetConfig

if TYPE_CHECKING:
    from litellm.proxy._types import NewProjectRequest, UpdateProjectRequest
    from litellm.proxy.utils import PrismaClient, ProxyLogging

_user_has_admin_view: Final = user_api_key_has_admin_view

# TODO: drop once the litellm-enterprise pin moves past 0.1.71, which imports this name
_is_user_team_admin: Final = is_team_admin


def validate_team_model_max_budget(
    model_max_budget: Mapping[str, BudgetConfig] | None,
    premium_user: bool,
) -> None:
    """Reject a team `model_max_budget` the limiter could not enforce (no duration, bad cap, tpm/rpm limits)."""
    if not model_max_budget:
        return
    if premium_user is not True:
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Setting model_max_budget on a team is an enterprise feature. {CommonProxyErrors.not_premium_user.value}"
            },
        )
    for model_name, budget_config in model_max_budget.items():
        if not model_name.strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "model_max_budget keys must be non-empty model names"},
            )
        max_budget = budget_config.max_budget
        if max_budget is None or not math.isfinite(max_budget) or max_budget < 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": (
                        f"model_max_budget[{model_name!r}].max_budget must be a non-negative finite number. "
                        f"Received: {max_budget}"
                    )
                },
            )
        if budget_config.budget_duration is None:
            raise HTTPException(
                status_code=400,
                detail={"error": f"model_max_budget[{model_name!r}] requires a budget_duration, e.g. '1d' or '30d'"},
            )
        validate_budget_duration(budget_config.budget_duration)
        if budget_config.tpm_limit is not None or budget_config.rpm_limit is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": (
                        f"model_max_budget[{model_name!r}] tpm_limit/rpm_limit are not enforced on a team; "
                        "set per-model rate limits on the key instead"
                    )
                },
            )


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
                    "Service-account keys cannot query user analytics. Use a user-bound key, or call as a proxy admin."
                )
            },
        )
    return user_api_key_dict.user_id


_ROUTE_LIST: Final = TypeAdapter(list[str] | None)


def _passthrough_routes_permission_error(field: str, entity: str) -> HTTPException:
    return HTTPException(
        status_code=403,
        detail={"error": f"Only proxy admins can set `{field}` on a {entity}."},
    )


def check_passthrough_routes_caller_permission(
    data: BaseModel | None,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    entity: str = "key",
    existing_metadata: Mapping[str, object] | None = None,
) -> None:
    """
    Only proxy admins may set `allowed_passthrough_routes` or `denied_passthrough_routes`
    (top-level or under `metadata`), since the runtime route checker reads both from key and
    team metadata.
    """
    check_allowed_passthrough_routes_caller_permission(data, user_api_key_dict, entity=entity)
    check_denied_passthrough_routes_caller_permission(
        data, user_api_key_dict, entity=entity, existing_metadata=existing_metadata
    )


_check_passthrough_routes_caller_permission: Final = check_passthrough_routes_caller_permission


def check_allowed_passthrough_routes_caller_permission(
    data: BaseModel | None,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    entity: str = "key",
) -> None:
    if data is None:
        return
    metadata: Final = getattr(data, "metadata", None)
    if isinstance(metadata, dict):
        try:
            _ROUTE_LIST.validate_python(metadata.get("denied_passthrough_routes"))
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "`metadata.denied_passthrough_routes` must be a list of route strings."},
            ) from e
    # view-only admins excluded by design; blocked upstream from writes anyway
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return
    if getattr(data, "allowed_passthrough_routes", None):
        raise _passthrough_routes_permission_error("allowed_passthrough_routes", entity)
    if isinstance(metadata, dict) and metadata.get("allowed_passthrough_routes"):
        raise _passthrough_routes_permission_error("metadata.allowed_passthrough_routes", entity)


def check_denied_passthrough_routes_caller_permission(
    data: BaseModel | None,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    entity: str = "key",
    existing_metadata: Mapping[str, object] | None = None,
) -> None:
    """
    A non-admin request must leave an existing deny list as it is: clearing it, or replacing
    `metadata` without it, would widen access. The outcome depends on the stored deny list, so
    run this only after the caller is known to be allowed to edit the object.
    """
    if data is None or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return
    metadata: Final = getattr(data, "metadata", None)
    existing_denied: Final = (existing_metadata or {}).get("denied_passthrough_routes") or None
    if (
        "denied_passthrough_routes" in data.model_fields_set
        and (getattr(data, "denied_passthrough_routes", None) or None) != existing_denied
    ):
        raise _passthrough_routes_permission_error("denied_passthrough_routes", entity)
    if _metadata_changes_denied_routes(data, metadata, existing_denied):
        raise _passthrough_routes_permission_error("metadata.denied_passthrough_routes", entity)


def _metadata_changes_denied_routes(data: BaseModel, metadata: object, existing_denied: object) -> bool:
    if isinstance(metadata, dict):
        return (metadata.get("denied_passthrough_routes") or None) != existing_denied
    return metadata is None and "metadata" in data.model_fields_set and existing_denied is not None


def check_disable_global_guardrails_caller_permission(
    disable_global_guardrails: bool | None,
    metadata: Mapping[str, object] | None,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    entity: str = "key",
    existing_metadata: Mapping[str, object] | None = None,
) -> None:
    """
    Only proxy admins may opt a key or team out of default-on guardrails, whether the
    flag is top-level or under `metadata`. Re-sending a flag that is already stored is
    not an opt-out, so non-admin edits of an already exempted object still go through.
    """
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return
    requested: Final = bool(disable_global_guardrails) or (
        metadata is not None and bool(metadata.get("disable_global_guardrails"))
    )
    if not requested:
        return
    if existing_metadata is not None and existing_metadata.get("disable_global_guardrails") is True:
        return
    raise HTTPException(
        status_code=403,
        detail={"error": f"Only proxy admins can set `disable_global_guardrails` on a {entity}."},
    )


_check_disable_global_guardrails_caller_permission: Final = check_disable_global_guardrails_caller_permission


def check_require_trace_id_caller_permission(
    require_trace_id: bool | None,
    metadata: Mapping[str, object] | None,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    metadata_sent: bool,
    existing_metadata: Mapping[str, object] | None = None,
) -> None:
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return
    stored: Final = existing_metadata is not None and existing_metadata.get("require_trace_id") is True
    metadata_value: Final = metadata is not None and metadata.get("require_trace_id") is True
    requested: Final = (
        require_trace_id if require_trace_id is not None else (metadata_value if metadata_sent else stored)
    )
    if requested is stored:
        return
    raise HTTPException(
        status_code=403,
        detail={"error": "Only proxy admins can set `require_trace_id` on a team."},
    )


def team_member_has_permission(
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


_team_member_has_permission: Final = team_member_has_permission


async def user_has_admin_privileges(
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
        user_obj: Final = await get_user_object(
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
            teams = await TeamRepository(prisma_client).table.find_many(where={"team_id": {"in": user_obj.teams}})

            for team in teams:
                team_obj = LiteLLM_TeamTable.model_validate(team.model_dump())
                if is_team_admin(user_api_key_dict=user_api_key_dict, team_obj=team_obj):
                    return True

    except Exception as e:
        # If there's an error checking, default to False for security
        verbose_proxy_logger.debug("Error checking admin privileges for user %s: %s", user_api_key_dict.user_id, e)
        return False

    return False


_user_has_admin_privileges: Final = user_has_admin_privileges


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
    admin_org_ids: Final = {
        m.organization_id
        for m in admin_user_obj.organization_memberships
        if m.user_role == LitellmUserRoles.ORG_ADMIN.value
    }
    if not admin_org_ids:
        return False
    if target_user_obj.organization_memberships is None:
        return False
    target_org_ids: Final = {m.organization_id for m in target_user_obj.organization_memberships}
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

    teams: Final = await TeamRepository(prisma_client).table.find_many(where={"team_id": {"in": admin_user_obj.teams}})
    admin_team_ids: Final = [
        team.team_id
        for team in teams
        if is_team_admin(
            user_api_key_dict=user_api_key_dict,
            team_obj=LiteLLM_TeamTable.model_validate(team.model_dump()),
        )
    ]
    if not admin_team_ids:
        return False
    target_team_ids: Final = set(target_user_obj.teams)
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
        cache: Final = user_api_key_cache or DualCacheImport()
        admin_user_obj: Final = await get_user_object(
            user_id=user_api_key_dict.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=cache,
            user_id_upsert=False,
            proxy_logging_obj=proxy_logging_obj,
        )
        if admin_user_obj is None:
            return False

        target_user_obj: Final = await get_user_object(
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
        verbose_proxy_logger.debug("Error checking invite permission for user %s: %s", user_api_key_dict.user_id, e)
        return False


def set_object_metadata_field(
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
        premium_user_check(field_name)

    object_data.metadata = object_data.metadata or {}
    object_data.metadata[field_name] = value


_set_object_metadata_field: Final = set_object_metadata_field


_TEAM_MEMBER_BUDGET_LIMIT_FIELDS: Final = (
    "max_budget",
    "soft_budget",
    "max_parallel_requests",
    "tpm_limit",
    "rpm_limit",
    "model_max_budget",
    "budget_duration",
    "allowed_models",
    "temp_budget_increase",
    "temp_budget_expiry",
)

_TEMP_BUDGET_FIELDS: Final = frozenset({"temp_budget_increase", "temp_budget_expiry"})


MEMBER_BUDGET_PATCH_FIELDS: Final = MappingProxyType(
    {
        "max_budget_in_team": "max_budget",
        "tpm_limit": "tpm_limit",
        "rpm_limit": "rpm_limit",
        "budget_duration": "budget_duration",
        "allowed_models": "allowed_models",
        "temp_budget_increase": "temp_budget_increase",
        "temp_budget_expiry": "temp_budget_expiry",
        "model_max_budget": "model_max_budget",
    }
)


_JSON_BUDGET_COLUMNS: Final = frozenset({"model_max_budget"})


def _prisma_value(column: str, value: object) -> object:
    if column in _JSON_BUDGET_COLUMNS:
        return json.dumps(value if value is not None else {})
    return list(value) if isinstance(value, tuple) else value


def _prisma_payload(data: Mapping[str, object]) -> dict[str, object]:
    return {column: _prisma_value(column, value) for column, value in data.items()}


def member_budget_patch(source: BaseModel) -> Mapping[str, object]:
    """Map the per-member limit fields a request actually set to their budget-table
    columns (merge-patch: a sent value updates, an explicit null clears, an absent
    field is left untouched)."""
    provided: Final = source.model_dump(exclude_unset=True)
    return {
        column: provided[request_field]
        for request_field, column in MEMBER_BUDGET_PATCH_FIELDS.items()
        if request_field in provided
    }


def _is_set_budget_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True


def _has_meaningful_budget_limit(budget_values: Mapping[str, object]) -> bool:
    """A budget is meaningful if at least one limit is actually set; an empty
    list (no model restriction) and None both count as unset."""
    return any(_is_set_budget_value(budget_values.get(field)) for field in _TEAM_MEMBER_BUDGET_LIMIT_FIELDS)


async def upsert_budget_and_membership(
    tx,
    *,
    team_id: str,
    user_id: str,
    existing_budget_id: str | None,
    user_api_key_dict: UserAPIKeyAuth,
    budget_patch: Mapping[str, object],
    team_default_budget_id: str | None = None,
    shared_budget_ids: frozenset[str] | None = None,
) -> None:
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

    ``shared_budget_ids`` extends that protection to any other row more than one
    membership points at, which a caller patching several members at once has
    already counted; a row listed there is cloned rather than written in place.
    A patch that only touches the temporary budget pair never copies permanent
    limits into a new row, so the member keeps inheriting the live team default.
    """
    ignore_default_model_budget_clear: Final = (
        existing_budget_id is not None
        and existing_budget_id == team_default_budget_id
        and "model_max_budget" in budget_patch
        and (
            budget_patch["model_max_budget"] is None
            or (isinstance(budget_patch["model_max_budget"], Mapping) and not budget_patch["model_max_budget"])
        )
    )
    effective_patch: Final = MappingProxyType(
        {
            field: value
            for field, value in budget_patch.items()
            if not (ignore_default_model_budget_clear and field == "model_max_budget")
        }
    )
    if not effective_patch:
        return

    write_data: Final = dict(effective_patch)
    if "budget_duration" in write_data:
        duration: Final = write_data["budget_duration"]
        write_data["budget_reset_at"] = (
            get_budget_reset_time(budget_duration=duration) if duration is not None else None
        )

    is_shared_default: Final = existing_budget_id is not None and (
        existing_budget_id == team_default_budget_id or existing_budget_id in (shared_budget_ids or frozenset())
    )
    temp_only: Final = frozenset(write_data) <= _TEMP_BUDGET_FIELDS

    async def _disconnect():
        await tx.litellm_teammembership.update(
            where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}},
            data={"litellm_budget_table": {"disconnect": True}},
        )

    if existing_budget_id is not None and not is_shared_default:
        existing_budget: Final = await tx.litellm_budgettable.find_unique(where={"budget_id": existing_budget_id})
        merged: Final = existing_budget.model_dump() if existing_budget is not None else {}
        merged.update(write_data)
        if not _has_meaningful_budget_limit(merged):
            await _disconnect()
            return
        await tx.litellm_budgettable.update(
            where={"budget_id": existing_budget_id},
            data={"updated_by": user_api_key_dict.user_id or "", **_prisma_payload(write_data)},
        )
        return

    source_row: Final = (
        await tx.litellm_budgettable.find_unique(where={"budget_id": existing_budget_id})
        if is_shared_default and not temp_only
        else None
    )
    source: Final[Mapping[str, object]] = source_row.model_dump() if source_row is not None else MappingProxyType({})

    create_data: Final[dict[str, object]] = {  # mutable-ok: Prisma create payloads are dict-shaped
        "created_by": user_api_key_dict.user_id or "",
        "updated_by": user_api_key_dict.user_id or "",
        "model_max_budget": {},
        **MappingProxyType(
            {
                f: source[f]
                for f in _TEAM_MEMBER_BUDGET_LIMIT_FIELDS
                if (
                    f != "model_max_budget"
                    or team_default_budget_id is None
                    or source.get("budget_id") != team_default_budget_id
                )
                and _is_set_budget_value(source.get(f))
            }
        ),
        **write_data,
    }

    # Restarting an inherited window on an unrelated edit hands the member a free period.
    carried: Final = source.get("budget_reset_at") if "budget_duration" not in effective_patch else None
    if carried is not None:
        create_data["budget_reset_at"] = carried
    if create_data.get("budget_reset_at") is None:
        create_data.pop("budget_reset_at", None)

    if not _has_meaningful_budget_limit(create_data):
        if existing_budget_id is not None and not temp_only:
            await _disconnect()
        return

    new_budget: Final = await tx.litellm_budgettable.create(
        data=_prisma_payload(create_data),
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


_upsert_budget_and_membership: Final = upsert_budget_and_membership


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
            premium_user_check()

    if field_name in updated_kv and updated_kv[field_name] is not None:
        # remove field from updated_kv
        _value: Final = updated_kv.pop(field_name)
        if "metadata" in updated_kv and updated_kv["metadata"] is not None:
            updated_kv["metadata"][field_name] = _value
        else:
            updated_kv["metadata"] = {field_name: _value}


def _has_non_empty_value(value: object) -> bool:
    """Check if a value has real content (not None, not empty list, not blank string)."""
    if value is None:
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def update_metadata_fields(updated_kv: dict) -> None:
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


_update_metadata_fields: Final = update_metadata_fields
