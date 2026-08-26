"""
Internal User Management Endpoints


These are members of a Team on LiteLLM

/user/new
/user/update
/user/bulk_update
/user/delete
/user/info
/user/list
"""

import asyncio
import json
import traceback
from collections.abc import Awaitable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final, Literal, cast

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import *
from litellm.proxy.auth.auth_checks import get_team_object, get_user_object
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import (
    object_permission_cache_key,
    user_object_permission_id_cache_key,
)
from litellm.proxy.hooks.model_max_budget_limiter import build_model_max_budget_usage
from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks
from litellm.proxy.management_endpoints.common_daily_activity import (
    DailySpendRecord,
    get_daily_activity,
    get_daily_activity_aggregated,
)
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_team_admin,
    _user_has_admin_view,
    require_caller_user_id_for_non_admin,
    validate_budget_duration,
    validate_finite_spend,
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _check_permissions_caller_permission,
    generate_key_helper_fn,
    prepare_metadata_fields,
)
from litellm.proxy.management_helpers.object_permission_utils import (
    _set_object_permission,
    handle_update_object_permission_common,
)
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.proxy.utils import handle_exception_on_proxy, hash_password
from litellm.repositories.organization_repository import OrganizationRepository
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import (
    InvitationLinkRepository,
    OrganizationMembershipRepository,
    TeamMembershipRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
    BulkUpdateUserRequest,
    BulkUpdateUserResponse,
    UserListResponse,
    UserUpdateResult,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIM_ENTERPRISE_METADATA_KEY,
    SCIM_ENTITLEMENTS_METADATA_KEY,
    SCIM_ROLES_METADATA_KEY,
)

if TYPE_CHECKING:
    from prisma import models as prisma_models
    from prisma import types as prisma_types

    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.proxy_server import PrismaClient
    from litellm.proxy.utils import ProxyLogging

router: Final = APIRouter()


def _user_table(
    prisma_client: "PrismaClient | None",
) -> "TableActions[prisma_models.LiteLLM_UserTable]":
    user_table: Final[TableActions[prisma_models.LiteLLM_UserTable]] = UserRepository(prisma_client).table
    return user_table


def _team_table(
    prisma_client: "PrismaClient | None",
) -> "TableActions[prisma_models.LiteLLM_TeamTable]":
    team_table: Final[TableActions[prisma_models.LiteLLM_TeamTable]] = TeamRepository(prisma_client).table
    return team_table


def _verification_token_table(
    prisma_client: "PrismaClient | None",
) -> "TableActions[prisma_models.LiteLLM_VerificationToken]":
    token_table: Final[TableActions[prisma_models.LiteLLM_VerificationToken]] = VerificationTokenRepository(
        prisma_client
    ).table
    return token_table


def _organization_membership_table(
    prisma_client: "PrismaClient | None",
) -> "TableActions[prisma_models.LiteLLM_OrganizationMembership]":
    membership_table: Final[TableActions[prisma_models.LiteLLM_OrganizationMembership]] = (
        OrganizationMembershipRepository(prisma_client).table
    )
    return membership_table


def _invitation_link_table(
    prisma_client: "PrismaClient | None",
) -> "TableActions[prisma_models.LiteLLM_InvitationLink]":
    invitation_table: Final[TableActions[prisma_models.LiteLLM_InvitationLink]] = InvitationLinkRepository(
        prisma_client
    ).table
    return invitation_table


def _organization_table(
    prisma_client: "PrismaClient | None",
) -> "TableActions[prisma_models.LiteLLM_OrganizationTable]":
    organization_table: Final[TableActions[prisma_models.LiteLLM_OrganizationTable]] = OrganizationRepository(
        prisma_client
    ).table
    return organization_table


def _team_membership_table(
    prisma_client: "PrismaClient | None",
) -> "TableActions[prisma_models.LiteLLM_TeamMembership]":
    team_membership_table: Final[TableActions[prisma_models.LiteLLM_TeamMembership]] = TeamMembershipRepository(
        prisma_client
    ).table
    return team_membership_table


def _hash_password_in_dict(data: dict) -> None:
    """Hash password field in-place if present."""
    if "password" in data and data["password"] is not None:
        data["password"] = hash_password(data["password"])


def _strip_password_from_response(response) -> None:
    """Strip password from API response (handles dicts, nested dicts, and Prisma models)."""
    if isinstance(response, dict):
        response.pop("password", None)
        if isinstance(response.get("data"), dict):
            response["data"].pop("password", None)
        elif hasattr(response.get("data"), "__dict__"):
            response["data"].__dict__.pop("password", None)


def _update_internal_new_user_params(data_json: dict, data: NewUserRequest) -> dict:
    if "user_id" in data_json and data_json["user_id"] is None:
        data_json["user_id"] = str(uuid.uuid4())

    auto_create_key: Final = data_json.pop("auto_create_key", True)

    if auto_create_key is False:
        data_json["table_name"] = "user"  # only create a user, don't create key if 'auto_create_key' set to False

    if litellm.default_internal_user_params and (
        data.user_role != LitellmUserRoles.PROXY_ADMIN.value and data.user_role != LitellmUserRoles.PROXY_ADMIN
    ):
        for key, value in litellm.default_internal_user_params.items():
            if key == "available_teams":
                continue
            elif (
                key not in data_json
                or data_json[key] is None
                or key == "models"
                and isinstance(data_json[key], list)
                and len(data_json[key]) == 0
            ):
                data_json[key] = value

    ## INTERNAL USER ROLE ONLY DEFAULT PARAMS ##
    if data.user_role is not None and data.user_role == LitellmUserRoles.INTERNAL_USER.value:
        if litellm.max_internal_user_budget is not None and data_json.get("max_budget") is None:
            data_json["max_budget"] = litellm.max_internal_user_budget

        if litellm.internal_user_budget_duration is not None and data_json.get("budget_duration") is None:
            data_json["budget_duration"] = litellm.internal_user_budget_duration

    data_json.pop("teams", None)  # handled separately
    return data_json


async def _check_duplicate_user_field(
    field_name: str,
    field_value: str | None,
    prisma_client: "PrismaClient | None",
    *,
    case_insensitive: bool = False,
    label: str | None = None,
) -> None:
    """
    Helper function to check if a field already exists in the user table.

    Args:
        field_name (str): Database field name to check.
        field_value (Optional[str]): Value to check for duplicates.
        prisma_client (Any): Database client instance.
        case_insensitive (bool): Whether to use case-insensitive comparison.
        label (Optional[str]): Human readable label for error messages.

    Raises:
        Exception: If database is not connected.
        HTTPException: If a user with the given field value already exists.
    """
    if field_value:
        if prisma_client is None:
            raise Exception("Database not connected")

        value: Final = field_value.strip()
        where_clause: Final = {field_name: {"equals": value}}
        if case_insensitive:
            where_clause[field_name]["mode"] = "insensitive"

        existing_user: Final[object] = await UserRepository(prisma_client).table.find_first(where=where_clause)

        if existing_user is not None:
            existing_value: Final = getattr(existing_user, field_name, value)
            error_label: Final = label or field_name
            raise HTTPException(
                status_code=409,
                detail={"error": f"User with {error_label} {existing_value} already exists"},
            )


async def _check_duplicate_user_email(user_email: str | None, prisma_client: "PrismaClient | None") -> None:
    """
    Helper function to check if a user email already exists in the database.
    """
    await _check_duplicate_user_field(
        field_name="user_email",
        field_value=user_email,
        prisma_client=prisma_client,
        case_insensitive=True,
        label="email",
    )


async def _check_duplicate_user_id(user_id: str | None, prisma_client: "PrismaClient | None") -> None:
    """
    Helper function to check if a user id already exists in the database.
    """
    await _check_duplicate_user_field(
        field_name="user_id",
        field_value=user_id,
        prisma_client=prisma_client,
        label="id",
    )


async def _add_user_to_organizations(
    user_id: str,
    organizations: list[str],
    prisma_client: "PrismaClient",
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Add a user to organizations
    """
    from litellm.proxy.management_endpoints.organization_endpoints import (
        organization_member_add,
    )

    tasks: Final[list[Awaitable[object]]] = []
    for organization_id in organizations:
        tasks.append(
            organization_member_add(
                data=OrganizationMemberAddRequest(
                    organization_id=organization_id,
                    member=[
                        OrgMember(
                            user_id=user_id,
                            role=LitellmUserRoles.INTERNAL_USER,
                        )
                    ],
                ),
                http_request=Request(
                    scope={"type": "http", "path": "/user/new"},
                ),
                user_api_key_dict=user_api_key_dict,
            )
        )
    await asyncio.gather(*tasks, return_exceptions=True)


async def _add_user_to_team(
    user_id: str,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    user_email: str | None = None,
    max_budget_in_team: float | None = None,
    user_role: Literal["user", "admin"] = "user",
):
    from litellm.proxy.management_endpoints.team_endpoints import team_member_add

    try:
        await team_member_add(
            data=TeamMemberAddRequest(
                team_id=team_id,
                member=Member(
                    user_id=user_id,
                    role=user_role,
                    user_email=user_email,
                ),
                max_budget_in_team=max_budget_in_team,
            ),
            user_api_key_dict=user_api_key_dict,
        )
    except HTTPException as e:
        if e.status_code == 400 and ("already exists" in str(e) or "doesn't exist" in str(e)):
            verbose_proxy_logger.debug(
                "litellm.proxy.management_endpoints.internal_user_endpoints.new_user(): User already exists in team - %s",
                e,
            )
        else:
            verbose_proxy_logger.error(
                "litellm.proxy.management_endpoints.internal_user_endpoints._add_user_to_team(): "
                "failed to add user %s to team %s - %s",
                user_id,
                team_id,
                str(e),
            )
    except Exception as e:
        if (
            "already exists" in str(e)
            or "doesn't exist" in str(e)
            or isinstance(e, ProxyException)
            and ProxyErrorTypes.team_member_already_in_team in e.type
        ):
            verbose_proxy_logger.debug(
                "litellm.proxy.management_endpoints.internal_user_endpoints.new_user(): User already exists in team - %s",
                e,
            )
        else:
            verbose_proxy_logger.error(
                "litellm.proxy.management_endpoints.internal_user_endpoints._add_user_to_team(): "
                "failed to add user %s to team %s - %s",
                user_id,
                team_id,
                str(e),
            )
            raise e


def check_if_default_team_set() -> list[str] | list[NewUserRequestTeam] | None:
    if litellm.default_internal_user_params is None:
        return None
    teams: Final = litellm.default_internal_user_params.get("teams")
    if teams is not None:
        if all(isinstance(team, str) for team in teams):
            return teams
        elif all(isinstance(team, dict) for team in teams):
            return [
                NewUserRequestTeam(
                    team_id=team.get("team_id"),
                    max_budget_in_team=team.get("max_budget_in_team"),
                    user_role=team.get("user_role", "user"),
                )
                for team in teams
            ]
        else:
            verbose_proxy_logger.error(
                "Invalid team type in default internal user params: %s",
                teams,
            )
    return None


async def add_new_user_to_default_team(
    user_id: str,
    user_email: str | None,
    user_api_key_dict: UserAPIKeyAuth,
    teams: list[str] | list[NewUserRequestTeam],
    prisma_client: "PrismaClient",
):
    tasks: Final[list[Awaitable[object]]] = []
    for team in teams:
        user_role: Literal["user", "admin"] = "user"
        max_budget_in_team: float | None = None
        if isinstance(team, str):
            team_id = team
        elif isinstance(team, NewUserRequestTeam):
            team_id = team.team_id
            user_role = team.user_role
            max_budget_in_team = team.max_budget_in_team
        else:
            raise ValueError(f"Invalid team type: {type(team)}")

        tasks.append(
            _add_user_to_team(
                user_id=user_id,
                team_id=team_id,
                user_email=user_email,
                user_api_key_dict=user_api_key_dict,
                max_budget_in_team=max_budget_in_team,
                user_role=user_role,
            )
        )
    await asyncio.gather(*tasks, return_exceptions=True)


@router.post(
    "/user/new",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=NewUserResponse,
)
@management_endpoint_wrapper
async def new_user(
    data: NewUserRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Use this to create a new INTERNAL user with a budget.
    Internal Users can access LiteLLM Admin UI to make keys, request access to models.
    This creates a new user and generates a new api key for the new user. The new api key is returned.

    Returns user id, budget + new key.

    Parameters:
    - user_id: Optional[str] - Specify a user id. If not set, a unique id will be generated.
    - user_alias: Optional[str] - A descriptive name for you to know who this user id refers to.
    - teams: Optional[list] - specify a list of team id's a user belongs to.
    - user_email: Optional[str] - Specify a user email.
    - send_invite_email: Optional[bool] - Specify if an invite email should be sent.
    - user_role: Optional[str] - Specify a user role - "proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer", "team", "customer". Info about each role here: `https://github.com/BerriAI/litellm/litellm/proxy/_types.py#L20`
    - max_budget: Optional[float] - Specify max budget for a given user.
    - budget_duration: Optional[str] - Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"), months ("1mo").
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models). Set to ['no-default-models'] to block all model access. Restricting user to only team-based model access.
    - tpm_limit: Optional[int] - Specify tpm limit for a given user (Tokens per minute)
    - rpm_limit: Optional[int] - Specify rpm limit for a given user (Requests per minute)
    - auto_create_key: bool - Default=True. Flag used for returning a key as part of the /user/new response
    - aliases: Optional[dict] - Model aliases for the user - [Docs](https://litellm.vercel.app/docs/proxy/virtual_keys#model-aliases)
    - config: Optional[dict] - [DEPRECATED PARAM] User-specific config.
    - allowed_cache_controls: Optional[list] - List of allowed cache control values. Example - ["no-cache", "no-store"]. See all values - https://docs.litellm.ai/docs/proxy/caching#turn-on--off-caching-per-request-
    - blocked: Optional[bool] - [Not Implemented Yet] Whether the user is blocked.
    - guardrails: Optional[List[str]] - [Not Implemented Yet] List of active guardrails for the user
    - policies: Optional[List[str]] - List of policy names to apply to the user. Policies define guardrails, conditions, and inheritance rules.
    - permissions: Optional[dict] - [Not Implemented Yet] User-specific permissions, eg. turning off pii masking.
    - metadata: Optional[dict] - Metadata for user, store information for user. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
    - max_parallel_requests: Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
    - soft_budget: Optional[float] - Get alerts when user crosses given budget, doesn't block requests.
    - model_max_budget: Optional[dict] - Model-specific max budget for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-budgets-to-keys)
    - budget_fallbacks: Optional[Dict[str, List[str]]] - Per-model fallback chain tried in order when that model's own `model_max_budget` is exceeded, e.g. {"gpt-4o": ["gpt-4o-mini"]}.
    - model_rpm_limit: Optional[float] - Model-specific rpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)
    - mcp_rpm_limit: Optional[dict] - Per-MCP-server rpm limit, keyed by MCP server name {"github": 100, "slack": 200}. Enforced for keys and teams only; values set on a user are stored but not enforced per user.
    - tag_rpm_limit: Optional[dict] - Per-request-tag rpm limit, keyed by request tag {"cell-1": 1000, "cell-2": 500}. Enforced for keys only; values set on a user are stored but not enforced per user.
    - model_tpm_limit: Optional[float] - Model-specific tpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)
    - spend: Optional[float] - Amount spent by user. Default is 0. Will be updated by proxy whenever user is used. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"), months ("1mo").
    - agent_id: Optional[str] - The agent id associated with the user.
    - team_id: Optional[str] - [DEPRECATED PARAM] The team id of the user. Default is None.
    - duration: Optional[str] - Duration for the key auto-created on `/user/new`. Default is None.
    - key_alias: Optional[str] - Alias for the key auto-created on `/user/new`. Default is None.
    - sso_user_id: Optional[str] - The id of the user in the SSO provider.
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - internal user-specific object permission. Example - {"vector_stores": ["vector_store_1"], "mcp_servers": ["github"], "mcp_tool_permissions": {"github": ["list_issues"]}}. The MCP grants act as a ceiling on every key this user holds. IF null or {} then no object permission.
    - prompts: Optional[List[str]] - List of allowed prompts for the user. If specified, the user will only be able to use these specific prompts.
    - organizations: List[str] - List of organization id's the user is a member of
    - budget_limits: Optional[list] - List of concurrent budget windows for the user. Each window specifies a budget_limit, time_period, and optional budget_duration. Example - [{"budget_limit": 10.0, "time_period": "1d"}, {"budget_limit": 50.0, "time_period": "7d"}].
    Returns:
    - key: (str) The generated api key for the user
    - expires: (datetime) Datetime object for when key expires.
    - user_id: (str) Unique user id - used for tracking spend across multiple keys for same user id.
    - max_budget: (float|None) Max budget for given user.

    Usage Example 

    ```shell
     curl -X POST "http://localhost:4000/user/new" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-1234" \
     -d '{
         "username": "new_user",
         "email": "new_user@example.com"
     }'
    ```
    """
    try:
        from litellm.proxy.proxy_server import _license_check, prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=400, detail=CommonProxyErrors.db_not_connected_error.value)

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail=CommonProxyErrors.db_not_connected_error.value,
            )
        validate_budget_duration(data.budget_duration)

        # Check for duplicate user_id or email
        await _check_duplicate_user_id(data.user_id, prisma_client)
        await _check_duplicate_user_email(data.user_email, prisma_client)

        # Check if license is over limit
        billable_users: Final = await UserRepository(prisma_client).count_billable_users()
        if billable_users and _license_check.is_over_limit(total_users=billable_users):
            raise HTTPException(
                status_code=403,
                detail="License is over limit. Please contact support@berri.ai to upgrade your license.",
            )

        # Only proxy admins can create administrative users
        # Check if user_api_key_dict is actually a UserAPIKeyAuth instance (not a Depends object)
        # This can happen when the function is called directly in tests
        if (
            data.user_role in [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY]
            and isinstance(user_api_key_dict, UserAPIKeyAuth)
            and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Only proxy admins can create administrative users (proxy_admin, proxy_admin_viewer). Attempted to create user with role: {data.user_role}. Your role: {user_api_key_dict.user_role}",
            )

        _check_permissions_caller_permission(
            data=data,
            user_api_key_dict=user_api_key_dict,
        )

        data_json = data.json()
        data_json = _update_internal_new_user_params(data_json, data)
        # Persist the requested grants as their own row and link it, mirroring key/team creation.
        # generate_key_helper_fn only forwards object_permission_id, so without this the entitlement
        # the caller sent would be dropped on the floor.
        data_json = await _set_object_permission(data_json=data_json, prisma_client=prisma_client)
        _hash_password_in_dict(data_json)
        teams = data.teams
        if teams is None:
            teams = check_if_default_team_set()
        organization_ids: Final = cast(list[str] | None, data_json.pop("organizations", None))

        response: Final = await generate_key_helper_fn(request_type="user", **data_json)
        # Admin UI Logic
        # Add User to Team and Organization
        # if team_id passed add this user to the team
        _team_id: Final = data_json.get("team_id", None)
        if _team_id is not None:
            await _add_user_to_team(
                user_id=cast(str, response.get("user_id")),
                team_id=_team_id,
                user_api_key_dict=user_api_key_dict,
                user_email=data.user_email,
                max_budget_in_team=None,
                user_role="user",
            )
        elif teams is not None:
            await add_new_user_to_default_team(
                user_id=cast(str, response.get("user_id")),
                user_email=data.user_email,
                user_api_key_dict=user_api_key_dict,
                teams=teams,
                prisma_client=prisma_client,
            )

        user_id: Final = cast(str | None, response.get("user_id", None))

        if organization_ids is not None and user_id is not None:
            await _add_user_to_organizations(
                user_id=user_id,
                organizations=organization_ids,
                prisma_client=prisma_client,
                user_api_key_dict=user_api_key_dict,
            )

        special_keys: Final = ["token", "token_id"]
        response_dict: Final = {}
        for key, value in response.items():
            if key in NewUserResponse.model_fields and key not in special_keys:
                response_dict[key] = value

        response_dict["key"] = response.get("token", "")

        new_user_response: Final = NewUserResponse.model_validate(response_dict)

        #########################################################
        ########## USER CREATED HOOK ################
        #########################################################
        asyncio.create_task(
            UserManagementEventHooks.async_user_created_hook(
                data=data,
                response=new_user_response,
                user_api_key_dict=user_api_key_dict,
            )
        )
        #########################################################
        ########## END USER CREATED HOOK ################
        #########################################################

        return new_user_response
    except Exception as e:
        verbose_proxy_logger.exception("/user/new: Exception occured - %s", e)
        raise handle_exception_on_proxy(e)


@router.get(
    "/user/available_roles",
    tags=["Internal User management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def ui_get_available_role(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Endpoint used by Admin UI to show all available roles to assign a user
    return {
        "proxy_admin": {
            "description": "Proxy Admin role",
            "ui_label": "Admin"
        }
    }
    """

    _data_to_return: Final = {}
    for role in LitellmUserRoles:
        # We only show a subset of roles on UI
        if role in [
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]:
            _data_to_return[role.value] = {
                "description": role.description,
                "ui_label": role.ui_label,
            }
    return _data_to_return


def get_team_from_list(
    team_list: list[LiteLLM_TeamTable] | list[TeamListResponseObject] | None,
    team_id: str,
) -> LiteLLM_TeamTable | LiteLLM_TeamMembership | None:
    if team_list is None:
        return None

    for team in team_list:
        if team.team_id == team_id:
            return team
    return None


def _is_valid_user_id(user_id: str) -> bool:
    """Validate that a decoded user_id is safe to use downstream."""
    MAX_USER_ID_LENGTH: Final = 512
    if len(user_id) > MAX_USER_ID_LENGTH:
        return False
    # Reject ASCII control characters (U+0000–U+001F)
    for ch in user_id:
        if ord(ch) < 0x20:
            return False
    return True


def get_user_id_from_request(request: Request) -> str | None:
    """
    Get the user id from the request
    """
    # Get the raw query string and parse it properly to handle + characters
    user_id: str | None = None
    query_string: Final = str(request.url.query)
    if "user_id=" in query_string:
        # Extract the user_id value from the raw query string
        import re
        from urllib.parse import unquote

        match: Final = re.search(r"user_id=([^&]*)", query_string)
        if match:
            # Use unquote instead of unquote_plus to preserve + characters
            raw_user_id: Final = unquote(match.group(1))
            if _is_valid_user_id(raw_user_id):
                user_id = raw_user_id
    return user_id


def _normalize_user_info_user_id(request: Request, user_id: str | None) -> str | None:
    """Normalize URL-decoded user_id while preserving '+' characters."""
    if user_id is not None and " " in user_id:
        return get_user_id_from_request(request=request)
    return user_id


def _enforce_user_info_access(user_id: str | None, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Re-validate that the caller may read the resolved ``user_id`` after
    URL-decoding has been finalized.

    The route-level check in ``RouteChecks.non_proxy_admin_allowed_routes_check``
    runs against ``request.query_params``, which decodes a literal ``+`` to a
    space. ``_normalize_user_info_user_id`` then re-parses the raw query with
    ``unquote`` so the endpoint can return rows for user_ids that contain ``+``
    (e.g. plus-addressed emails). That asymmetry let an attacker who registered
    a username with a literal space pass the route check and then read another
    user's row by sending the encoded ``+`` form. Re-checking ownership here
    closes the gap without changing the supported user_id grammar.
    """
    if user_id is None:
        return
    # Admin-view roles (PROXY_ADMIN and PROXY_ADMIN_VIEW_ONLY) bypass
    # ownership, mirroring the `/user/info` carve-out that
    # `RouteChecks.non_proxy_admin_allowed_routes_check` applies upstream.
    if _user_has_admin_view(user_api_key_dict):
        return
    if user_id == user_api_key_dict.user_id:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            f"key not allowed to access this user's info. user_id={user_id}, key's user_id={user_api_key_dict.user_id}"
        ),
    )


async def _get_user_info_teams(
    prisma_client: Any,
    user_id: str | None,
    user_info: Any | None,
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[list[TeamListResponseObject], list[TeamListResponseObject] | None]:
    """Fetch and merge teams from membership + user.teams field."""
    from litellm.proxy.management_endpoints.team_endpoints import list_team

    team_list: list[TeamListResponseObject] = []
    team_id_list: list[str] = []

    teams_1: Final = await list_team(
        http_request=Request(
            scope={"type": "http", "path": "/user/info"},
        ),
        user_id=user_id,
        user_api_key_dict=user_api_key_dict,
    )

    if teams_1 is not None and isinstance(teams_1, list):
        team_list = teams_1
        team_id_list = [team.team_id for team in teams_1]

    teams_2: list[TeamListResponseObject] | None = None
    target_team_ids: Final = getattr(user_info, "teams", None)

    if target_team_ids and isinstance(target_team_ids, list):
        teams_2 = await prisma_client.get_data(
            team_id_list=target_team_ids,
            table_name="team",
            query_type="find_all",
        )
    elif user_api_key_dict.user_id is not None and user_id is None:
        caller_user_info: Final[object] = await prisma_client.get_data(user_id=user_api_key_dict.user_id)
        caller_team_ids: Final = getattr(caller_user_info, "teams", None)
        if caller_team_ids:
            teams_2 = await prisma_client.get_data(
                team_id_list=caller_team_ids,
                table_name="team",
                query_type="find_all",
            )

    if teams_2 is not None and isinstance(teams_2, list):
        for team in teams_2:
            if team.team_id not in team_id_list:
                team_list.append(team)
                team_id_list.append(team.team_id)

    return team_list, teams_1


_SCIM_DIRECTORY_METADATA_KEYS: Final = frozenset(
    {SCIM_ENTERPRISE_METADATA_KEY, SCIM_ENTITLEMENTS_METADATA_KEY, SCIM_ROLES_METADATA_KEY}
)


def _redact_scim_enterprise_metadata(
    metadata: dict[str, object] | None,
) -> dict[str, object] | None:
    """SCIM enterprise attributes, entitlements, and roles are persisted in user
    metadata so reporting can group on them, but they are directory-only fields
    that generic user-info endpoints must not surface; SCIM clients read them
    through the SCIM endpoints."""
    if not isinstance(metadata, dict) or not _SCIM_DIRECTORY_METADATA_KEYS.intersection(metadata):
        return metadata
    return {k: v for k, v in metadata.items() if k not in _SCIM_DIRECTORY_METADATA_KEYS}


def _build_user_info_response(
    user_id: str | None,
    user_info: Any | None,
    keys: list[LiteLLM_VerificationToken] | None,
    team_list: list[TeamListResponseObject],
    teams_1: list[TeamListResponseObject] | None,
    model_max_budget_usage: dict[str, dict[str, object]] | None = None,
) -> UserInfoResponse:
    """Create UserInfoResponse while filtering sensitive fields."""
    if user_info is None and keys is not None:
        spend: Final = sum(getattr(k, "spend", 0) for k in keys)
        user_info = {"spend": spend}

    returned_keys: Final = _process_keys_for_user_info(keys=keys, all_teams=teams_1)
    team_list.sort(key=lambda x: getattr(x, "team_alias", "") or "")

    _user_info: Final = user_info.model_dump() if isinstance(user_info, BaseModel) else user_info
    if isinstance(_user_info, dict):
        _user_info.pop("password", None)
        _user_info["metadata"] = _redact_scim_enterprise_metadata(_user_info.get("metadata"))
        if model_max_budget_usage is not None:
            _user_info["model_max_budget_usage"] = model_max_budget_usage

    return UserInfoResponse(
        user_id=user_id,
        user_info=_user_info,
        keys=returned_keys,
        teams=team_list,
    )


@router.get(
    "/user/info",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UserInfoResponse,
)
@management_endpoint_wrapper
async def user_info(
    request: Request,
    user_id: str | None = fastapi.Query(default=None, description="User ID in the request parameters"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [10/07/2024]
    Note: To get all users (+pagination), use `/user/list` endpoint.


    Use this to get user information. (user row + all user key info)

    Example request
    ```
    curl -X GET 'http://localhost:4000/user/info?user_id=krrish7%40berri.ai' \
    --header 'Authorization: Bearer sk-1234'
    ```
    """
    from litellm.proxy.proxy_server import model_max_budget_limiter, prisma_client

    try:
        user_id = _normalize_user_info_user_id(request=request, user_id=user_id)
        _enforce_user_info_access(user_id=user_id, user_api_key_dict=user_api_key_dict)

        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        if user_id is None and _user_has_admin_view(user_api_key_dict):
            return await _get_user_info_for_proxy_admin(user_api_key_dict=user_api_key_dict)
        elif user_id is None:
            user_id = user_api_key_dict.user_id
        ## GET USER ROW ##

        user_info = None
        if user_id is not None:
            user_info = await prisma_client.get_data(user_id=user_id)

        if user_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found",
            )

        team_list, teams_1 = await _get_user_info_teams(
            prisma_client=prisma_client,
            user_id=user_id,
            user_info=user_info,
            user_api_key_dict=user_api_key_dict,
        )

        ## GET ALL KEYS ##
        keys: Final = await prisma_client.get_data(
            user_id=user_id,
            table_name="key",
            query_type="find_all",
        )

        response_data: Final = _build_user_info_response(
            user_id=user_id,
            user_info=user_info,
            keys=keys,
            team_list=team_list,
            teams_1=teams_1,
            model_max_budget_usage=await build_model_max_budget_usage(
                entity_type=Litellm_EntityType.USER,
                entity_id=user_id,
                model_max_budget=getattr(user_info, "model_max_budget", None),
                cache=model_max_budget_limiter.dual_cache,
            ),
        )

        return response_data
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.user_info(): Exception occured - %s", e)
        raise handle_exception_on_proxy(e)


async def _check_user_info_v2_access(
    user_api_key_dict: UserAPIKeyAuth,
    target_user_id: str,
) -> "prisma_models.LiteLLM_UserTable | None":
    """
    Check if the caller is allowed to access the target user's info.

    Returns the target user's DB row if access is allowed, None otherwise.
    Returning the row avoids a redundant DB fetch in the caller.

    Access rules:
    1. Proxy admins / proxy admin viewers can access any user
    2. User can access their own info
    3. Team admins can access info of users in their teams

    Raises on unexpected DB errors so they surface as 500s, not silent 404s.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None

    # Helper: fetch the target user row (reused across branches). object_permission is included so
    # callers can read the user's MCP/vector-store entitlements without a second round trip.
    async def _fetch_target_user():
        return await _user_table(prisma_client).find_unique(
            where={"user_id": target_user_id}, include={"object_permission": True}
        )

    # Rule 1: Proxy admins — fetch and return the target row directly
    if _user_has_admin_view(user_api_key_dict):
        return await _fetch_target_user()

    # Rule 2: Self-lookup
    if user_api_key_dict.user_id == target_user_id:
        return await _fetch_target_user()

    # Rule 3: Team admins can look up users in their teams
    if user_api_key_dict.user_id is not None:
        # Get caller's teams
        caller_user: Final = await _user_table(prisma_client).find_unique(where={"user_id": user_api_key_dict.user_id})
        if caller_user is not None and caller_user.teams:
            # Fetch the target user ONCE, before the loop
            target_user: Final = await _fetch_target_user()
            if target_user is None:
                return None

            # Get all teams the caller belongs to
            teams: Final = await _team_table(prisma_client).find_many(where={"team_id": {"in": caller_user.teams}})
            for team in teams:
                team_obj = LiteLLM_TeamTable.model_validate(team.model_dump())
                if _is_user_team_admin(user_api_key_dict=user_api_key_dict, team_obj=team_obj):
                    # Check if target user is in this team
                    if team.team_id in (target_user.teams or []):
                        return target_user

    return None


@router.get(
    "/v2/user/info",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UserInfoV2Response,
)
@management_endpoint_wrapper
async def user_info_v2(
    request: Request,
    user_id: str | None = fastapi.Query(default=None, description="User ID in the request parameters"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Lightweight endpoint to get user info. Returns only the user object — no keys, no teams objects.

    This is the v2 replacement for /user/info, designed to avoid the "god endpoint" problem
    where the old endpoint loaded all keys and teams into memory.

    Access control:
    - Proxy admins can query any user
    - Team admins can query users within their teams
    - Internal users can only query themselves (omit user_id or pass own)
    - Returns 404 for non-existent users or unauthorized access

    Example request:
    ```
    curl -X GET 'http://localhost:4000/v2/user/info?user_id=user123' \\
    --header 'Authorization: Bearer sk-1234'
    ```
    """
    from litellm.proxy.proxy_server import model_max_budget_limiter, prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail=CommonProxyErrors.db_not_connected_error.value,
            )

        # Handle URL encoding for + characters
        if user_id is not None and " " in user_id:
            user_id = get_user_id_from_request(request=request)

        # Default to self-lookup if no user_id provided
        if user_id is None:
            user_id = user_api_key_dict.user_id

        if user_id is None:
            raise HTTPException(
                status_code=400,
                detail="user_id is required. Either pass it as a query parameter or authenticate with a user-bound key.",
            )

        # Check access — returns the user row if allowed, None otherwise.
        # This avoids a redundant DB fetch since the access check already
        # loads the target user for team-admin verification.
        user_row: Final = await _check_user_info_v2_access(
            user_api_key_dict=user_api_key_dict,
            target_user_id=user_id,
        )

        if user_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"User not found: {user_id}",
            )

        user_data: Final = user_row.model_dump()

        return UserInfoV2Response(
            user_id=user_data.get("user_id", user_id),
            user_email=user_data.get("user_email"),
            user_alias=user_data.get("user_alias"),
            user_role=user_data.get("user_role"),
            spend=user_data.get("spend", 0.0),
            max_budget=user_data.get("max_budget"),
            models=user_data.get("models") or [],
            budget_duration=user_data.get("budget_duration"),
            budget_reset_at=user_data.get("budget_reset_at"),
            metadata=_redact_scim_enterprise_metadata(user_data.get("metadata")),
            created_at=user_data.get("created_at"),
            updated_at=user_data.get("updated_at"),
            sso_user_id=user_data.get("sso_user_id"),
            teams=user_data.get("teams") or [],
            object_permission=user_data.get("object_permission"),
            model_max_budget=user_data.get("model_max_budget"),
            model_max_budget_usage=await build_model_max_budget_usage(
                entity_type=Litellm_EntityType.USER,
                entity_id=user_data.get("user_id", user_id),
                model_max_budget=user_data.get("model_max_budget"),
                cache=model_max_budget_limiter.dual_cache,
            ),
        )
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.user_info_v2(): Exception occured - %s", e)
        raise handle_exception_on_proxy(e)


async def _get_user_info_for_proxy_admin(user_api_key_dict: UserAPIKeyAuth):
    """
    Admin UI Endpoint - Returns All Teams and Keys when Proxy Admin is querying

    - get all teams in LiteLLM_TeamTable
    - get all keys in LiteLLM_VerificationToken table

    Why separate helper for proxy admin ?
        - To get Faster UI load times, get all teams and virtual keys in 1 query
    """

    from litellm.proxy.proxy_server import prisma_client

    sql_query: Final = """
        SELECT 
            (SELECT json_agg(t.*) FROM "LiteLLM_TeamTable" t) as teams,
            (SELECT json_agg(k.*) FROM "LiteLLM_VerificationToken" k WHERE k.team_id != 'litellm-dashboard' OR k.team_id IS NULL) as keys
    """
    if prisma_client is None:
        raise Exception(
            "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
        )

    results: Final = await prisma_client.db.query_raw(sql_query)

    verbose_proxy_logger.debug("results_keys: %s", results)

    _keys_in_db: Final[Sequence[dict[str, object]]] = results[0]["keys"] or []
    # cast all keys to LiteLLM_VerificationToken
    keys_in_db: Final = []
    for key in _keys_in_db:
        if key.get("models") is None:
            key["models"] = []
        keys_in_db.append(LiteLLM_VerificationToken.model_validate(key))

    # cast all teams to LiteLLM_TeamTable
    _teams_in_db: list[LiteLLM_TeamTable] = results[0]["teams"] or []
    _teams_in_db = [LiteLLM_TeamTable.model_validate(team) for team in _teams_in_db]
    _teams_in_db.sort(key=lambda x: getattr(x, "team_alias", "") or "")
    returned_keys: Final = _process_keys_for_user_info(keys=keys_in_db, all_teams=_teams_in_db)

    # Get admin's own user_id and user_info
    admin_user_id: Final = user_api_key_dict.user_id
    admin_user_info = None

    if admin_user_id is not None:
        admin_user_info = await prisma_client.get_data(user_id=admin_user_id)
        if admin_user_info is not None:
            admin_user_info = (
                admin_user_info.model_dump() if isinstance(admin_user_info, BaseModel) else admin_user_info
            )
            if isinstance(admin_user_info, dict):
                admin_user_info.pop("password", None)

    return UserInfoResponse(
        user_id=admin_user_id,
        user_info=admin_user_info,
        keys=returned_keys,
        teams=_teams_in_db,
    )


def _process_keys_for_user_info(
    keys: list[LiteLLM_VerificationToken] | None,
    all_teams: list[LiteLLM_TeamTable] | list[TeamListResponseObject] | None,
):
    from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
    from litellm.proxy.proxy_server import general_settings, litellm_master_key_hash

    returned_keys: Final = []
    if keys is None:
        pass
    else:
        for key in keys:
            if (
                key.token == litellm_master_key_hash
                and general_settings.get("disable_master_key_return", False)
                is True  ## [IMPORTANT] used by hosted proxy-ui to prevent sharing master key on ui
            ):
                continue

            try:
                _key: dict = key.model_dump()
            except Exception:
                # if using pydantic v1
                _key = key.dict()

            # Filter out UI session tokens (team_id="litellm-dashboard")
            if _key.get("team_id") == UI_SESSION_TOKEN_TEAM_ID:
                continue

            if "team_id" in _key and _key["team_id"] is not None and _key["team_id"] != "litellm-dashboard":
                team_info = get_team_from_list(team_list=all_teams, team_id=_key["team_id"])
                if team_info is not None:
                    team_alias = getattr(team_info, "team_alias", None)
                    _key["team_alias"] = team_alias
                else:
                    _key["team_alias"] = None
            else:
                _key["team_alias"] = "None"
            returned_keys.append(_key)
    return returned_keys


def _update_internal_user_params(data_json: dict, data: UpdateUserRequest | UpdateUserRequestNoUserIDorEmail) -> dict:
    non_default_values: Final = {}
    fields_set: Final = data.fields_set() if hasattr(data, "fields_set") else set()

    for k, v in data_json.items():
        if k == "max_budget":
            if "max_budget" in fields_set:
                non_default_values[k] = v
        elif (
            v is not None
            and v
            not in (
                [],
                {},
            )
            and k not in LiteLLM_ManagementEndpoint_MetadataFields
        ):  # models default to [], spend defaults to 0, we should not reset these values
            non_default_values[k] = v

    is_internal_user = False
    if data.user_role == LitellmUserRoles.INTERNAL_USER:
        is_internal_user = True

    if "budget_duration" in non_default_values:
        from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

        validate_budget_duration(non_default_values["budget_duration"])
        non_default_values["budget_reset_at"] = get_budget_reset_time(
            budget_duration=non_default_values["budget_duration"]
        )

    if "max_budget" not in non_default_values:
        if (
            is_internal_user and litellm.max_internal_user_budget is not None
        ):  # applies internal user limits, if user role updated
            non_default_values["max_budget"] = litellm.max_internal_user_budget

    if "budget_duration" not in non_default_values:  # applies internal user limits, if user role updated
        if is_internal_user and litellm.internal_user_budget_duration is not None:
            non_default_values["budget_duration"] = litellm.internal_user_budget_duration
            from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

            non_default_values["budget_reset_at"] = get_budget_reset_time(
                budget_duration=non_default_values["budget_duration"]
            )

    return non_default_values


async def _schedule_user_update_audit_log(
    response: dict[str, Any],
    existing_user_row: BaseModel | None,
    litellm_changed_by: str | None,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str | None,
) -> None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return
    try:
        updated_user_row: Final = await _user_table(prisma_client).find_first(where={"user_id": response["user_id"]})
        if updated_user_row:
            user_row_typed: Final = LiteLLM_UserTable.model_validate(updated_user_row.model_dump(exclude_none=True))
            asyncio.create_task(
                UserManagementEventHooks.create_internal_user_audit_log(
                    user_id=user_row_typed.user_id,
                    action="updated",
                    litellm_changed_by=litellm_changed_by or user_api_key_dict.user_id,
                    user_api_key_dict=user_api_key_dict,
                    litellm_proxy_admin_name=litellm_proxy_admin_name,
                    before_value=(existing_user_row.model_dump_json(exclude_none=True) if existing_user_row else None),
                    after_value=user_row_typed.model_dump_json(exclude_none=True),
                )
            )
    except Exception as audit_error:
        verbose_proxy_logger.warning("Failed to create audit log for user %s: %s", response.get("user_id"), audit_error)


def _check_user_update_authz(
    user_request: UpdateUserRequest,
    user_api_key_dict: UserAPIKeyAuth,
    existing_user_row: BaseModel | None,
) -> None:
    """Authorization checks for /user/update — raises HTTPException on failure."""
    if user_request.user_role is not None and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        raise HTTPException(status_code=403, detail="Only proxy admins can modify user roles.")

    if existing_user_row is not None:
        typed_row: Final = LiteLLM_UserTable.model_validate(existing_user_row.model_dump(exclude_none=True))
        if not can_user_call_user_update(user_api_key_dict=user_api_key_dict, user_info=typed_row):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "User does not have permission to update this user. Only PROXY_ADMIN can update other users."
                },
            )
    elif user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        # Silent-create guard: only PROXY_ADMIN may create via /user/update.
        raise HTTPException(
            status_code=404,
            detail={
                "error": "User not found. Only PROXY_ADMIN can create users via /user/update; use /user/new instead."
            },
        )


async def _invalidate_user_spend_counter_if_changed(
    non_default_values: Mapping[str, object],
) -> None:
    """Invalidate the cross-pod spend counter after a direct ``spend`` change.

    A direct ``spend`` change must also invalidate the cross-pod spend counter
    enforcement reads; the DB write alone leaves a warm counter at the stale
    value. ``non_default_values["user_id"]`` is populated in every branch of the
    caller (incl. the email-new-user insert path, whose response is a bare model
    and not safely subscriptable).
    """
    if non_default_values.get("spend") is not None:
        from litellm.proxy.proxy_server import _invalidate_spend_counter

        await _invalidate_spend_counter(counter_key=f"spend:user:{non_default_values['user_id']}")


def _clears_object_permission(user_request: UpdateUserRequest) -> bool:
    """Whether the caller explicitly asked to remove this user's object_permission.

    Distinguishes "sent nothing" from "sent an empty grant set". Only the latter clears; an omitted
    field must leave an existing entitlement alone.
    """
    if "object_permission" not in (user_request.fields_set() if hasattr(user_request, "fields_set") else set()):
        return False
    sent: Final = user_request.object_permission
    return sent is None or not sent.model_dump(exclude_unset=True, exclude_none=True)


async def _invalidate_cached_user_entitlement(user_id: str | None, object_permission_ids: tuple[str, ...]) -> None:
    """Drop the cache entries an entitlement change makes stale.

    All three kinds are needed: a permission row is cached under its own id (so re-reading the same
    link still yields the OLD grants), the ``user_id -> object_permission_id`` link is cached
    separately (so a user who previously had NO entitlement keeps its "none" sentinel), and the user
    row itself is cached whole. Leaving any behind means an admin revoking a tool keeps serving it
    until the management-object TTL expires.

    Both the outgoing and incoming permission ids are passed, because a clear leaves no incoming id
    at all and an upsert may mint a new row; invalidating only one of the two leaves the other's
    grants live.

    Each deletion is isolated: one that fails must not skip the others, or a single unreachable key
    would silently leave the rest of a revocation in place. Best-effort overall, exactly as the caches
    are everywhere else, since one we cannot clear still expires on its own.
    """
    from litellm.proxy.proxy_server import user_api_key_cache

    keys: Final = (
        *(object_permission_cache_key(permission_id) for permission_id in dict.fromkeys(object_permission_ids)),
        *((user_object_permission_id_cache_key(user_id), user_id) if user_id is not None else ()),
    )
    for key in keys:
        try:
            await user_api_key_cache.async_delete_cache(key=key)
        except Exception as e:  # noqa: BLE001  # a cache we cannot clear still expires; never fail the write
            verbose_proxy_logger.warning("Failed to invalidate cached entitlement key %r: %s", key, e)


async def _update_single_user_helper(
    user_request: UpdateUserRequest,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: str | None = None,
) -> dict[str, Any]:
    """
    Helper function to update a single user.
    Used by both user_update and bulk_user_update endpoints.

    Returns the updated user data or raises an exception on failure.
    """
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise Exception("Not connected to DB!")

    if not user_request.user_id and not user_request.user_email:
        raise ValueError("Either user_id or user_email must be provided")

    _check_permissions_caller_permission(
        data=user_request,
        user_api_key_dict=user_api_key_dict,
    )

    data_json: Final[dict] = user_request.model_dump(exclude_unset=True)
    non_default_values = _update_internal_user_params(data_json=data_json, data=user_request)
    _hash_password_in_dict(non_default_values)

    existing_user_row: BaseModel | None = None
    if user_request.user_id:
        existing_user_row = await _user_table(prisma_client).find_first(where={"user_id": user_request.user_id})
    elif user_request.user_email:
        existing_user_row = await _user_table(prisma_client).find_first(where={"user_email": user_request.user_email})

    _check_user_update_authz(user_request, user_api_key_dict, existing_user_row)

    if existing_user_row is not None:
        existing_user_row = LiteLLM_UserTable.model_validate(existing_user_row.model_dump(exclude_none=True))

    # Prevent budget self-escalation (GHSA-wvg4-6222-3q4r): non-admin callers
    # must not be able to raise their own budget/spend fields.
    # can_user_call_user_update() already restricts non-admins to self-updates,
    # so this guard only fires for self-escalation attempts.
    _target_user_id: Final = user_request.user_id or (
        getattr(existing_user_row, "user_id", None) if existing_user_row is not None else None
    )
    _is_self_update: Final = _target_user_id is not None and user_api_key_dict.user_id == _target_user_id
    if _is_self_update and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        # object_permission is a CEILING on what this human may reach, so a self-write is an
        # escalation path: sending an empty grant list means "no restriction" and would lift a
        # restriction an admin placed on them. Checked against the fields the caller actually SENT,
        # because `_update_internal_user_params` drops empty values, and `object_permission: {}` is
        # precisely the clear-my-own-ceiling case this must refuse.
        _sent_fields: Final = user_request.fields_set() if hasattr(user_request, "fields_set") else set()
        _protected_fields: Final = ("max_budget", "soft_budget", "spend", "object_permission")
        for _field in _protected_fields:
            if _field in non_default_values or _field in _sent_fields:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": f"Non-admin users cannot modify '{_field}' on their own record. Contact your proxy admin."
                    },
                )

    existing_metadata: Final = (
        cast(dict, getattr(existing_user_row, "metadata", {}) or {}) if existing_user_row is not None else {}
    )

    non_default_values = prepare_metadata_fields(
        data=user_request,
        non_default_values=non_default_values,
        existing_metadata=existing_metadata or {},
    )

    # Reject NaN/±inf spend before it can reach the DB / spend counter.
    validate_finite_spend(non_default_values.get("spend"))

    # Upsert the grants into their own row and link it, mirroring /key/update and /team/update.
    # This also removes object_permission from the payload, which is not a column on the user table.
    if "object_permission" in non_default_values:
        object_permission_id: Final = await handle_update_object_permission_common(
            data_json=non_default_values,
            existing_object_permission_id=getattr(existing_user_row, "object_permission_id", None),
            prisma_client=prisma_client,
        )
        if object_permission_id is not None:
            non_default_values["object_permission_id"] = object_permission_id
    elif _clears_object_permission(user_request):
        # An explicit `{}` or null means "no object permission", which the merge-based upsert cannot
        # express: merging an empty grant set over the existing row leaves every grant in place. So
        # the link is dropped instead, which is what makes the documented clear actually clear.
        non_default_values["object_permission_id"] = None

    # Perform the update
    response: dict[str, Any] | None = None

    if user_request.user_id and len(user_request.user_id) > 0:
        non_default_values["user_id"] = user_request.user_id
        response = await prisma_client.update_data(
            user_id=user_request.user_id,
            data=non_default_values,
            table_name="user",
        )
    elif user_request.user_email:
        # Handle email-based updates
        existing_user_rows: Final = await prisma_client.get_data(
            key_val={"user_email": user_request.user_email},
            table_name="user",
            query_type="find_all",
        )

        if existing_user_rows and isinstance(existing_user_rows, list) and len(existing_user_rows) > 0:
            for existing_user in existing_user_rows:
                non_default_values["user_id"] = existing_user.user_id
                response = await prisma_client.update_data(
                    user_id=existing_user.user_id,
                    data=non_default_values,
                    table_name="user",
                )
                break  # Update first matching user
        else:
            # Create new user if not found
            non_default_values["user_id"] = str(uuid.uuid4())
            non_default_values["user_email"] = user_request.user_email
            inserted_user_row: Final = await prisma_client.insert_data(data=non_default_values, table_name="user")
            response = inserted_user_row  # pyright: ignore[reportAssignmentType]  # insert_data returns a prisma row

    if response is not None:
        await _schedule_user_update_audit_log(
            response=response,
            existing_user_row=existing_user_row,
            litellm_changed_by=litellm_changed_by,
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
        )

        await _invalidate_user_spend_counter_if_changed(non_default_values)

        if "object_permission_id" in non_default_values:
            await _invalidate_cached_user_entitlement(
                user_id=non_default_values.get("user_id"),
                object_permission_ids=tuple(
                    permission_id
                    for permission_id in (
                        getattr(existing_user_row, "object_permission_id", None),
                        non_default_values.get("object_permission_id"),
                    )
                    if isinstance(permission_id, str)
                ),
            )

    if response is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Failed to update user"},
        )
    _strip_password_from_response(response)
    return response


def can_user_call_user_update(
    user_api_key_dict: UserAPIKeyAuth,
    user_info: LiteLLM_UserTable,
) -> bool:
    """
    Helper to check if the user has access to the key's info
    """
    if (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
        or user_api_key_dict.user_id == user_info.user_id
    ):
        return True
    return False


@router.post(
    "/user/update",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def user_update(
    data: UpdateUserRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Example curl 

    ```
    curl --location 'http://0.0.0.0:4000/user/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "test-litellm-user-4",
        "user_role": "proxy_admin_viewer"
    }'
    ```
    
    Parameters:
        - user_id: Optional[str] - Specify a user id. If not set, a unique id will be generated.
        - user_email: Optional[str] - Specify a user email.
        - password: Optional[str] - Specify a user password.
        - user_alias: Optional[str] - A descriptive name for you to know who this user id refers to.
        - teams: Optional[list] - specify a list of team id's a user belongs to.
        - send_invite_email: Optional[bool] - Specify if an invite email should be sent.
        - user_role: Optional[str] - Specify a user role - "proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer", "team", "customer". Info about each role here: `https://github.com/BerriAI/litellm/litellm/proxy/_types.py#L20`
        - max_budget: Optional[float] - Specify max budget for a given user.
        - budget_duration: Optional[str] - Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"), months ("1mo").
        - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
        - tpm_limit: Optional[int] - Specify tpm limit for a given user (Tokens per minute)
        - rpm_limit: Optional[int] - Specify rpm limit for a given user (Requests per minute)
        - auto_create_key: bool - Default=True. Flag used for returning a key as part of the /user/new response
        - aliases: Optional[dict] - Model aliases for the user - [Docs](https://litellm.vercel.app/docs/proxy/virtual_keys#model-aliases)
        - config: Optional[dict] - [DEPRECATED PARAM] User-specific config.
        - allowed_cache_controls: Optional[list] - List of allowed cache control values. Example - ["no-cache", "no-store"]. See all values - https://docs.litellm.ai/docs/proxy/caching#turn-on--off-caching-per-request-
        - blocked: Optional[bool] - [Not Implemented Yet] Whether the user is blocked.
        - guardrails: Optional[List[str]] - [Not Implemented Yet] List of active guardrails for the user
        - policies: Optional[List[str]] - List of policy names to apply to the user. Policies define guardrails, conditions, and inheritance rules.
        - permissions: Optional[dict] - [Not Implemented Yet] User-specific permissions, eg. turning off pii masking.
        - metadata: Optional[dict] - Metadata for user, store information for user. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
        - max_parallel_requests: Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
        - soft_budget: Optional[float] - Get alerts when user crosses given budget, doesn't block requests.
        - model_max_budget: Optional[dict] - Model-specific max budget for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-budgets-to-keys)
        - budget_fallbacks: Optional[Dict[str, List[str]]] - Per-model fallback chain tried in order when that model's own `model_max_budget` is exceeded, e.g. {"gpt-4o": ["gpt-4o-mini"]}.
        - model_rpm_limit: Optional[float] - Model-specific rpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)
        - mcp_rpm_limit: Optional[dict] - Per-MCP-server rpm limit, keyed by MCP server name {"github": 100, "slack": 200}. Enforced for keys and teams only; values set on a user are stored but not enforced per user.
        - tag_rpm_limit: Optional[dict] - Per-request-tag rpm limit, keyed by request tag {"cell-1": 1000, "cell-2": 500}. Enforced for keys only; values set on a user are stored but not enforced per user.
        - model_tpm_limit: Optional[float] - Model-specific tpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)
        - spend: Optional[float] - Amount spent by user. Default is 0. Will be updated by proxy whenever user is used. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"), months ("1mo").
        - agent_id: Optional[str] - The agent id associated with the user.
        - team_id: Optional[str] - [DEPRECATED PARAM] The team id of the user. Default is None.
        - duration: Optional[str] - [NOT IMPLEMENTED].
        - key_alias: Optional[str] - [NOT IMPLEMENTED].
        - object_permission: Optional[LiteLLM_ObjectPermissionBase] - internal user-specific object permission. Example - {"vector_stores": ["vector_store_1"], "mcp_servers": ["github"], "mcp_tool_permissions": {"github": ["list_issues"]}}. The MCP grants act as a ceiling on every key this user holds. IF null or {} then no object permission.
        - prompts: Optional[List[str]] - List of allowed prompts for the user. If specified, the user will only be able to use these specific prompts.
        - budget_limits: Optional[list] - List of concurrent budget windows for the user. Each window specifies a budget_limit, time_period, and optional budget_duration. Example - [{"budget_limit": 10.0, "time_period": "1d"}, {"budget_limit": 50.0, "time_period": "7d"}].

    """
    try:
        verbose_proxy_logger.debug("/user/update: Received data = %s", data)

        response: Final = await _update_single_user_helper(
            user_request=data,
            user_api_key_dict=user_api_key_dict,
        )
        return response
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.user_update(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({e})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


async def bulk_update_processed_users(
    users_to_update: list[UpdateUserRequest],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: str | None = None,
) -> BulkUpdateUserResponse:
    results: Final[list[UserUpdateResult]] = []
    successful_updates = 0
    failed_updates = 0

    # Process each user update independently
    try:
        for user_request in users_to_update:
            try:
                response = await _update_single_user_helper(
                    user_request=user_request,
                    user_api_key_dict=user_api_key_dict,
                    litellm_changed_by=litellm_changed_by,
                )
                # Record success
                results.append(
                    UserUpdateResult(
                        user_id=(response.get("user_id") if response else user_request.user_id),
                        user_email=user_request.user_email,
                        success=True,
                        updated_user=response,
                    )
                )
                successful_updates += 1
            except Exception as e:
                verbose_proxy_logger.exception(
                    "Failed to update user %s: %s", user_request.user_id or user_request.user_email, e
                )
                # Record failure
                error_message = str(e)
                verbose_proxy_logger.error(
                    "Failed to update user %s: %s", user_request.user_id or user_request.user_email, error_message
                )

                results.append(
                    UserUpdateResult(
                        user_id=user_request.user_id,
                        user_email=user_request.user_email,
                        success=False,
                        error=error_message,
                    )
                )
                failed_updates += 1

        return BulkUpdateUserResponse(
            results=results,
            total_requested=len(users_to_update),
            successful_updates=successful_updates,
            failed_updates=failed_updates,
        )
    except Exception as e:
        verbose_proxy_logger.exception("Failed to update users: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/user/bulk_update",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=BulkUpdateUserResponse,
)
@management_endpoint_wrapper
async def bulk_user_update(
    data: BulkUpdateUserRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: str | None = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Bulk update multiple users at once.
    
    This endpoint allows updating multiple users in a single request. Each user update
    is processed independently - if some updates fail, others will still succeed.
    
    Parameters:
    - users: Optional[List[UpdateUserRequest]] - List of specific user update requests
    - all_users: Optional[bool] - Set to true to update all users in the system
    - user_updates: Optional[UpdateUserRequest] - Updates to apply when all_users=True
    
    Returns:
    - results: List of individual update results
    - total_requested: Total number of users requested for update
    - successful_updates: Number of successful updates
    - failed_updates: Number of failed updates
    
    Example request for specific users:
    ```bash
    curl --location 'http://0.0.0.0:4000/user/bulk_update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "users": [
            {
                "user_id": "user1",
                "user_role": "internal_user",
                "max_budget": 100.0
            },
            {
                "user_email": "user2@example.com", 
                "user_role": "internal_user_viewer",
                "max_budget": 50.0
            }
        ]
    }'
    ```
    
    Example request for all users:
    ```bash
    curl --location 'http://0.0.0.0:4000/user/bulk_update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "all_users": true,
        "user_updates": {
            "user_role": "internal_user",
            "max_budget": 50.0
        }
    }'
    ```
    """
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected"},
        )

    # Only proxy admins can modify user_role in bulk updates
    _bulk_role = getattr(data.user_updates, "user_role", None) if data.user_updates else None
    if _bulk_role is None and data.users:
        _bulk_role = next((u.user_role for u in data.users if u.user_role is not None), None)
    if _bulk_role is not None and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        raise HTTPException(
            status_code=403,
            detail="Only proxy admins can modify user roles.",
        )

    # Determine the list of users to update
    users_to_update: list[UpdateUserRequest] | list[UpdateUserRequestNoUserIDorEmail] = []

    if data.all_users and data.user_updates:
        # Only proxy admins can update all users at once
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value:
            raise HTTPException(
                status_code=403,
                detail="Only proxy admins can update all users at once.",
            )
        # Optimized path for updating all users directly in database
        all_users_in_db: Final = await _user_table(prisma_client).find_many(order={"created_at": "desc"})

        if not all_users_in_db:
            raise HTTPException(
                status_code=400,
                detail={"error": "No users found to update"},
            )

        # Limit batch size to prevent overwhelming the system
        MAX_BATCH_SIZE = 500  # Increased limit for all-users operations
        if len(all_users_in_db) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Maximum {MAX_BATCH_SIZE} users can be updated at once. Found {len(all_users_in_db)} users."
                },
            )

        # Apply update transformations (reuse existing logic)
        data_json: Final[dict] = data.user_updates.model_dump(exclude_unset=True)
        non_default_values: Final[dict[str, object]] = _update_internal_user_params(
            data_json=data_json, data=data.user_updates
        )

        # Remove user identification fields since we're updating by user_id
        non_default_values.pop("user_id", None)
        non_default_values.pop("user_email", None)

        successful_updates = 0
        failed_updates: Final = 0
        results: Final[list[UserUpdateResult]] = []

        try:
            # Perform bulk database update
            await UserRepository(prisma_client).table.update_many(
                where={},
                data=non_default_values,  # Update all users
            )

            # Create individual success results
            for user in all_users_in_db:
                results.append(
                    UserUpdateResult(
                        user_id=user.user_id,
                        user_email=user.user_email,
                        success=True,
                        updated_user={"user_id": user.user_id, **non_default_values},
                    )
                )
                successful_updates += 1

            # Create single audit log entry for bulk operation
            try:
                asyncio.create_task(
                    UserManagementEventHooks.create_internal_user_audit_log(
                        user_id=user_api_key_dict.user_id or "",
                        action="updated",
                        litellm_changed_by=litellm_changed_by or user_api_key_dict.user_id,
                        user_api_key_dict=user_api_key_dict,
                        litellm_proxy_admin_name=litellm_proxy_admin_name,
                        before_value=f"Updated {len(all_users_in_db)} users",
                        after_value=json.dumps(non_default_values),
                    )
                )
            except Exception as audit_error:
                verbose_proxy_logger.warning("Failed to create bulk audit log: %s", audit_error)

        except Exception as e:
            verbose_proxy_logger.exception("Failed to perform bulk update: %s", e)
            # Fall back to individual updates if bulk update fails
            for user in all_users_in_db:
                user_update_request = data.user_updates.model_copy()
                user_update_request.user_id = user.user_id
                users_to_update.append(user_update_request)

        if successful_updates > 0:
            return BulkUpdateUserResponse(
                results=results,
                total_requested=len(all_users_in_db),
                successful_updates=successful_updates,
                failed_updates=failed_updates,
            )

    elif data.users:
        users_to_update = data.users
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Must specify either 'users' for individual updates or 'all_users=True' with 'user_updates' for bulk updates"
            },
        )

    if not users_to_update:
        raise HTTPException(
            status_code=400,
            detail={"error": "No users found to update"},
        )

    # Limit batch size to prevent overwhelming the system
    MAX_BATCH_SIZE = 500  # Increased limit for all-users operations
    if len(users_to_update) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Maximum {MAX_BATCH_SIZE} users can be updated at once. Found {len(users_to_update)} users."
            },
        )

    return await bulk_update_processed_users(
        users_to_update=cast(list[UpdateUserRequest], users_to_update),
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=litellm_changed_by,
    )


async def get_user_key_counts(
    prisma_client: "PrismaClient | None",
    user_ids: list[str] | None = None,
) -> Mapping[str, int]:
    """
    Helper function to get the count of keys for each user using Prisma's count method.

    Args:
        prisma_client: The Prisma client instance
        user_ids: List of user IDs to get key counts for

    Returns:
        Dictionary mapping user_id to key count
    """
    from litellm.constants import UI_SESSION_TOKEN_TEAM_ID

    if not user_ids or len(user_ids) == 0:
        return {}

    result: Final[dict[str, int]] = {}

    # Get count for each user_id individually
    for user_id in user_ids:
        count = await _verification_token_table(prisma_client).count(
            where={
                "user_id": user_id,
                "OR": [
                    {"team_id": None},
                    {"team_id": {"not": UI_SESSION_TOKEN_TEAM_ID}},
                ],
            }
        )
        result[user_id] = count

    return result


def _validate_sort_params(sort_by: str | None, sort_order: str) -> dict[str, str] | None:
    order_by: Final[dict[str, str]] = {}

    if sort_by is None:
        return None
    # Validate sort_by is a valid column
    valid_columns: Final = [
        "user_id",
        "user_email",
        "created_at",
        "spend",
        "user_alias",
        "user_role",
    ]
    if sort_by not in valid_columns:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid sort column. Must be one of: {', '.join(valid_columns)}"},
        )

    # Validate sort_order
    if sort_order.lower() not in ["asc", "desc"]:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid sort order. Must be 'asc' or 'desc'"},
        )

    order_by[sort_by] = sort_order.lower()

    return order_by


async def _authorize_user_list_request(
    user_api_key_dict: UserAPIKeyAuth,
    organization_ids: str | None,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: "UserApiKeyCache",
    proxy_logging_obj: "ProxyLogging | None",
) -> str | None:
    """
    Authorize the /user/list request and return the (possibly scoped) organization_ids string.

    - Proxy admins: returns organization_ids unchanged (may be None).
    - Org admins: returns comma-separated org IDs scoped to their allowed orgs.
    - Others: raises 403.
    """
    if _user_has_admin_view(user_api_key_dict):
        return organization_ids

    if user_api_key_dict.user_id is None:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins and organization admins can list users."},
        )
    try:
        caller_user: Final = await get_user_object(
            user_id=user_api_key_dict.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            proxy_logging_obj=proxy_logging_obj,
        )
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins and organization admins can list users."},
        )
    if caller_user is None:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins and organization admins can list users."},
        )

    allowed_org_ids = [
        m.organization_id
        for m in (caller_user.organization_memberships or [])
        if m.user_role == LitellmUserRoles.ORG_ADMIN.value
    ]
    if not allowed_org_ids:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins and organization admins can list users."},
        )

    # If client also sent organization_ids, intersect with allowed orgs
    if organization_ids:
        requested: Final = set(oid.strip() for oid in organization_ids.split(",") if oid.strip())
        intersection: Final = list(requested & set(allowed_org_ids))
        if not intersection:
            raise HTTPException(
                status_code=403,
                detail={"error": "You do not have org_admin access to the requested organization(s)."},
            )
        allowed_org_ids = intersection

    return ",".join(allowed_org_ids)


@router.get(
    "/user/list",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UserListResponse,
)
async def get_users(
    role: str | None = fastapi.Query(default=None, description="Filter users by role"),
    user_ids: str | None = fastapi.Query(default=None, description="Get list of users by user_ids"),
    sso_user_ids: str | None = fastapi.Query(default=None, description="Get list of users by sso_user_id"),
    user_email: str | None = fastapi.Query(default=None, description="Filter users by partial email match"),
    team: str | None = fastapi.Query(default=None, description="Filter users by team id"),
    page: int = fastapi.Query(default=1, ge=1, description="Page number"),
    page_size: int = fastapi.Query(default=25, ge=1, le=100, description="Number of items per page"),
    sort_by: str | None = fastapi.Query(
        default=None,
        description="Column to sort by (e.g. 'user_id', 'user_email', 'created_at', 'spend')",
    ),
    sort_order: str = fastapi.Query(default="asc", description="Sort order ('asc' or 'desc')"),
    organization_ids: str | None = fastapi.Query(
        default=None,
        description="Filter users by organization membership. Comma-separated list of org IDs.",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a paginated list of users with filtering and sorting options.

    Parameters:
        role: Optional[str]
            Filter users by role. Can be one of:
            - proxy_admin
            - proxy_admin_viewer
            - internal_user
            - internal_user_viewer
        user_ids: Optional[str]
            Get list of users by user_ids. Comma separated list of user_ids.
        sso_ids: Optional[str]
            Get list of users by sso_ids. Comma separated list of sso_ids.
        user_email: Optional[str]
            Filter users by partial email match
        team: Optional[str]
            Filter users by team id. Will match if user has this team in their teams array.
        page: int
            The page number to return
        page_size: int
            The number of items per page
        sort_by: Optional[str]
            Column to sort by (e.g. 'user_id', 'user_email', 'created_at', 'spend')
        sort_order: Optional[str]
            Sort order ('asc' or 'desc')
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": f"No db connected. prisma client={prisma_client}"},
        )

    # Server-side authorization: proxy admins see all, org admins see only their org(s)
    organization_ids = await _authorize_user_list_request(
        user_api_key_dict=user_api_key_dict,
        organization_ids=organization_ids,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    # Calculate skip and take for pagination
    skip: Final = (page - 1) * page_size

    # Build where conditions based on provided parameters
    where_conditions: dict[str, object] = {}

    if role:
        where_conditions["user_role"] = role

    if user_ids and isinstance(user_ids, str):
        user_id_list: Final = [uid.strip() for uid in user_ids.split(",") if uid.strip()]
        if len(user_id_list) == 1:
            where_conditions["user_id"] = {
                "contains": user_id_list[0],
                "mode": "insensitive",
            }
        else:
            where_conditions["user_id"] = {
                "in": user_id_list,
            }

    if user_email is not None and isinstance(user_email, str):
        where_conditions["user_email"] = {
            "contains": user_email,
            "mode": "insensitive",  # Case-insensitive search
        }

    if team is not None and isinstance(team, str):
        where_conditions["teams"] = {
            "has": team  # Array contains for string arrays in Prisma
        }

    if sso_user_ids is not None and isinstance(sso_user_ids, str):
        sso_id_list: Final = [sid.strip() for sid in sso_user_ids.split(",") if sid.strip()]
        where_conditions["sso_user_id"] = {
            "in": sso_id_list,
        }

    if organization_ids:
        org_id_list: Final = [oid.strip() for oid in organization_ids.split(",") if oid.strip()]
        if org_id_list:
            where_conditions["organization_memberships"] = {"some": {"organization_id": {"in": org_id_list}}}

    ## Filter any none fastapi.Query params - e.g. where_conditions: {'user_email': {'contains': Query(None), 'mode': 'insensitive'}, 'teams': {'has': Query(None)}}
    where_conditions = {k: v for k, v in where_conditions.items() if v is not None}

    # Build order_by conditions

    order_by: Final[dict[str, str] | None] = (
        _validate_sort_params(sort_by, sort_order) if sort_by is not None and isinstance(sort_by, str) else None
    )

    users: Final[Sequence[prisma_models.LiteLLM_UserTable]] = await UserRepository(prisma_client).table.find_many(
        where=where_conditions,
        skip=skip,
        take=page_size,
        order=(order_by if order_by else {"created_at": "desc"}),  # Default to created_at desc if no sort specified
    )

    # Get total count of user rows
    total_count: Final[int] = await UserRepository(prisma_client).table.count(where=where_conditions)

    # Get key count for each user
    user_key_counts: Final = await get_user_key_counts(prisma_client, [user.user_id for user in users])

    verbose_proxy_logger.debug("Total count of users: %s", total_count)

    # Calculate total pages
    total_pages: Final = -(-total_count // page_size)  # Ceiling division

    # Prepare response
    user_list: list[LiteLLM_UserTableWithKeyCount] = []
    for user in users:
        user_dump = user.model_dump()
        user_dump["metadata"] = _redact_scim_enterprise_metadata(user_dump.get("metadata"))
        user_list.append(
            LiteLLM_UserTableWithKeyCount.model_validate(
                {**user_dump, "key_count": user_key_counts.get(user.user_id, 0)}
            )
        )

    return {
        "users": user_list,
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@router.post(
    "/user/delete",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def delete_user(
    data: DeleteUserRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: str | None = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    delete user and associated user keys

    ```
    curl --location 'http://0.0.0.0:4000/user/delete' \

    --header 'Authorization: Bearer sk-1234' \

    --header 'Content-Type: application/json' \

    --data-raw '{
        "user_ids": ["45e3e396-ee08-4a61-a88e-16b3ce7e0849"]
    }'
    ```

    Parameters:
    - user_ids: List[str] - The list of user id's to be deleted.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        _cleanup_members_with_roles,
    )
    from litellm.proxy.management_helpers.audit_logs import (
        get_audit_log_changed_by,
        is_audit_logging_enabled,
    )
    from litellm.proxy.proxy_server import (
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.user_ids is None:
        raise HTTPException(status_code=400, detail={"error": "No user id passed in"})

    # Per-target authorization: the route-level gate accepts this call when
    # the caller is PROXY_ADMIN or an ORG_ADMIN of *any* org named in
    # request_data["organization_id"]/["organizations"]. That gate does NOT
    # cross-check data.user_ids against the caller's scope, so without this
    # loop an org-admin of org-A could delete users in org-B by supplying
    # {"user_ids": [victim_in_org_B], "organization_id": "org-A"}.
    caller_is_proxy_admin: Final = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    caller_admin_org_ids: set[str] = set()
    if not caller_is_proxy_admin:
        caller_memberships: Final[Sequence[prisma_models.LiteLLM_OrganizationMembership]] = (
            await _organization_membership_table(prisma_client).find_many(
                where={
                    "user_id": user_api_key_dict.user_id,
                    "user_role": LitellmUserRoles.ORG_ADMIN.value,
                }
            )
            if user_api_key_dict.user_id
            else []
        )
        caller_admin_org_ids = {m.organization_id for m in caller_memberships if m.organization_id}
        if not caller_admin_org_ids:
            raise HTTPException(
                status_code=403,
                detail={"error": "Only PROXY_ADMIN or ORG_ADMIN users may delete users."},
            )

    # Batch-fetch target memberships once before the per-user loop. Avoids
    # an N+1 DB call when delete_user is called with a large user_ids list.
    target_org_ids_by_user: Final[dict[str, set[str]]] = {}
    if not caller_is_proxy_admin:
        all_target_memberships: Final = await _organization_membership_table(prisma_client).find_many(
            where={"user_id": {"in": data.user_ids}}
        )
        for m in all_target_memberships:
            if not m.organization_id:
                continue
            target_org_ids_by_user.setdefault(m.user_id, set()).add(m.organization_id)

    # check that all teams passed exist
    for user_id in data.user_ids:
        user_row = await UserRepository(prisma_client).table.find_unique(where={"user_id": user_id})

        if user_row is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"User not found, passed user_id={user_id}"},
            )

        if not caller_is_proxy_admin:
            target_org_ids = target_org_ids_by_user.get(user_id, set())
            # Org-admin may only delete users whose entire org membership is
            # within their admin scope. A target with ANY org outside the
            # caller's scope (or no org at all) requires PROXY_ADMIN.
            if not target_org_ids or not target_org_ids.issubset(caller_admin_org_ids):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": (
                            f"User {user_id} is not within your admin scope. "
                            "Only PROXY_ADMIN may delete users outside your "
                            "administered organizations."
                        )
                    },
                )

        # we do this after the first for loop, since first for loop is for validation. we only want this inserted after validation passes
        if is_audit_logging_enabled():
            # make an audit log for each team deleted
            _user_row = user_row.model_dump_json(exclude_none=True)

            asyncio.create_task(
                create_audit_log_for_update(
                    request_data=LiteLLM_AuditLogs(
                        id=str(uuid.uuid4()),
                        updated_at=datetime.now(timezone.utc),
                        changed_by=get_audit_log_changed_by(
                            litellm_changed_by=litellm_changed_by,
                            user_api_key_dict=user_api_key_dict,
                            litellm_proxy_admin_name=litellm_proxy_admin_name,
                        ),
                        changed_by_api_key=user_api_key_dict.api_key,
                        table_name=LitellmTableNames.USER_TABLE_NAME,
                        object_id=user_id,
                        action="deleted",
                        updated_values="{}",
                        before_value=_user_row,
                    )
                )
            )

        ## CLEANUP MEMBERS_WITH_ROLES
        fetch_all_teams: Sequence[prisma_models.LiteLLM_TeamTable] = await TeamRepository(
            prisma_client
        ).table.find_many(where={"team_id": {"in": user_row.teams}})
        teams_to_update: list[tuple[str, str]] = []
        for team in fetch_all_teams:
            removed_team_members, new_team_members = _cleanup_members_with_roles(
                existing_team_row=LiteLLM_TeamTable.model_validate(team.model_dump()),
                data=TeamMemberDeleteRequest(
                    team_id=team.team_id,
                    user_id=user_row.user_id,
                    user_email=user_row.user_email,
                ),
            )
            if removed_team_members:
                _db_new_team_members: list[dict] = [m.model_dump() for m in new_team_members]
                teams_to_update.append((team.team_id, json.dumps(_db_new_team_members)))

        ## update teams

        for team_id, members_with_roles in teams_to_update:
            await TeamRepository(prisma_client).table.update(
                where={"team_id": team_id},
                data={"members_with_roles": members_with_roles},
            )
    # End of Audit logging

    ## DELETE ASSOCIATED KEYS
    await _verification_token_table(prisma_client).delete_many(where={"user_id": {"in": data.user_ids}})

    ## DELETE ASSOCIATED INVITATION LINKS
    await _invitation_link_table(prisma_client).delete_many(
        where={
            "OR": [
                {"user_id": {"in": data.user_ids}},
                {"created_by": {"in": data.user_ids}},
                {"updated_by": {"in": data.user_ids}},
            ]
        }
    )

    ## DELETE ASSOCIATED ORGANIZATION MEMBERSHIPS
    await _organization_membership_table(prisma_client).delete_many(where={"user_id": {"in": data.user_ids}})

    ## DELETE ASSOCIATED TEAM MEMBERSHIPS
    await _team_membership_table(prisma_client).delete_many(where={"user_id": {"in": data.user_ids}})

    ## DELETE USERS
    deleted_users: Final = await _user_table(prisma_client).delete_many(where={"user_id": {"in": data.user_ids}})

    return deleted_users


async def add_internal_user_to_organization(
    user_id: str,
    organization_id: str,
    user_role: LitellmUserRoles,
) -> "prisma_models.LiteLLM_OrganizationMembership":
    """
    Helper function to add an internal user to an organization

    Adds the user to LiteLLM_OrganizationMembership table

    - Checks if organization_id exists

    Raises:
    - Exception if database not connected
    - Exception if user_id or organization_id not found
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise Exception("Database not connected")

    try:
        # Check if organization_id exists
        organization_row: Final = await _organization_table(prisma_client).find_unique(
            where={"organization_id": organization_id}
        )
        if organization_row is None:
            raise Exception(f"Organization not found, passed organization_id={organization_id}")

        # Create a new organization membership entry
        new_membership: Final[prisma_models.LiteLLM_OrganizationMembership] = await OrganizationMembershipRepository(
            prisma_client
        ).table.create(
            data={
                "user_id": user_id,
                "organization_id": organization_id,
                "user_role": user_role,
                # Note: You can also set budget within an organization if needed
            }
        )

        return new_membership
    except Exception as e:
        raise Exception(f"Failed to add user to organization: {e}")


async def _resolve_org_filter_for_user_search(
    user_api_key_dict: UserAPIKeyAuth,
    team_id: str | None,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: "UserApiKeyCache",
    proxy_logging_obj: "ProxyLogging | None",
) -> list[str] | None:
    """
    Return a list of org IDs to filter by, or ``None`` for no filter.

    Reads the ``scope_user_search_to_org`` UI-setting flag and applies
    role-based access rules when the flag is ON.
    """
    from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
        get_ui_settings_cached,
    )

    ui_settings: Final = await get_ui_settings_cached()
    if not ui_settings.get("scope_user_search_to_org", False):
        return None  # flag OFF — no filtering

    if _user_has_admin_view(user_api_key_dict):
        return None  # proxy admin — see everything

    # Try to resolve org admin memberships
    caller_user = None
    if user_api_key_dict.user_id is not None:
        try:
            caller_user = await get_user_object(
                user_id=user_api_key_dict.user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_id_upsert=False,
                proxy_logging_obj=proxy_logging_obj,
            )
        except ValueError:
            caller_user = None

    # Collect org IDs from ALL org memberships (any role, not just ORG_ADMIN).
    # This allows team admins who are org members to search users in their org.
    member_org_ids: list[str] = []
    if caller_user is not None:
        member_org_ids = [m.organization_id for m in (caller_user.organization_memberships or [])]

    if member_org_ids:
        return member_org_ids

    # Fall back to resolving via team_id (query param or from the caller's API key)
    resolved_team_id: Final = team_id or user_api_key_dict.team_id
    if resolved_team_id is not None:
        return await _resolve_team_org_filter(
            user_api_key_dict,
            resolved_team_id,
            prisma_client,
            user_api_key_cache,
            proxy_logging_obj,
        )

    raise HTTPException(
        status_code=403,
        detail={
            "error": "scope_user_search_to_org is enabled. Only proxy admins, organization admins, or team admins can search users."
        },
    )


async def _resolve_team_org_filter(
    user_api_key_dict: UserAPIKeyAuth,
    team_id: str,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: "UserApiKeyCache",
    proxy_logging_obj: "ProxyLogging | None",
) -> list[str]:
    """Look up the team and return its org as a filter list, or raise 403."""
    from litellm.proxy.management_endpoints.common_utils import _is_user_team_admin

    try:
        team_obj: Final = await get_team_object(
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except HTTPException:
        raise HTTPException(
            status_code=403,
            detail={"error": f"scope_user_search_to_org is enabled but team '{team_id}' was not found."},
        )

    if not _is_user_team_admin(user_api_key_dict, team_obj):
        raise HTTPException(
            status_code=403,
            detail={"error": "scope_user_search_to_org is enabled. You must be an admin of this team to search users."},
        )

    if team_obj.organization_id:
        return [team_obj.organization_id]

    raise HTTPException(
        status_code=403,
        detail={
            "error": "scope_user_search_to_org is enabled and this team is not part of an organization. Contact your proxy admin to adjust this setting."
        },
    )


@router.get(
    "/user/filter/ui",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
    responses={
        200: {"model": list[LiteLLM_UserTableFiltered]},
    },
)
async def ui_view_users(
    user_id: str | None = fastapi.Query(default=None, description="User ID in the request parameters"),
    user_email: str | None = fastapi.Query(default=None, description="User email in the request parameters"),
    team_id: str | None = fastapi.Query(
        default=None,
        description="Team ID — used when a team admin searches for users to add to their team",
    ),
    page: int = fastapi.Query(default=1, description="Page number for pagination", ge=1),
    page_size: int = fastapi.Query(default=50, description="Number of items per page", ge=1, le=100),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Filter users based on partial match of user_id or email with pagination.

    Behaviour depends on the ``scope_user_search_to_org`` UI-setting flag
    (stored in the ``litellm_uisettings`` table):

    * **Flag OFF (default):** any authenticated user can search all users.
    * **Flag ON:**
      - Proxy admins see all users.
      - Org admins see only users in their org(s).
      - Team admins for an org-bound team see users in that org.
      - Others receive a 403.
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    try:
        org_filter_ids: Final = await _resolve_org_filter_for_user_search(
            user_api_key_dict=user_api_key_dict,
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Calculate offset for pagination
        skip: Final = (page - 1) * page_size

        # Build where conditions based on provided parameters
        where_conditions: Final[prisma_types.LiteLLM_UserTableWhereInput] = {}

        if user_id:
            where_conditions["user_id"] = {
                "contains": user_id,
                "mode": "insensitive",  # Case-insensitive search
            }

        if user_email:
            where_conditions["user_email"] = {
                "contains": user_email,
                "mode": "insensitive",  # Case-insensitive search
            }

        # Apply org filter when scope_user_search_to_org is ON and caller is not proxy admin
        if org_filter_ids is not None:
            where_conditions["organization_memberships"] = {"some": {"organization_id": {"in": org_filter_ids}}}

        # Query users with pagination and filters
        users: Final = await _user_table(prisma_client).find_many(
            where=where_conditions,
            skip=skip,
            take=page_size,
            order={"created_at": "desc"},
        )

        if not users:
            return []

        return [LiteLLM_UserTableFiltered.model_validate(user.model_dump()) for user in users]

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error searching users: %s", e)
        raise HTTPException(status_code=500, detail=f"Error searching users: {e}")


# Using shared metric helper implementations from common_daily_activity


async def _resolve_user_email_metadata(
    prisma_client: "PrismaClient", records: Sequence[DailySpendRecord]
) -> dict[str, dict]:
    """Map each user_id on the page to its email/alias so the Usage dashboard can
    label the 'Spend Per User' chart with the email instead of the raw UUID."""
    user_ids: Final = {
        user_id for record in records if isinstance(user_id := getattr(record, "user_id", None), str) and user_id
    }
    if not user_ids:
        return {}
    users: Final = await _user_table(prisma_client).find_many(where={"user_id": {"in": list(user_ids)}})
    return {user.user_id: {"user_email": user.user_email, "user_alias": user.user_alias} for user in users}


@router.get(
    "/user/daily/activity",
    tags=["Budget & Spend Tracking", "Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SpendAnalyticsPaginatedResponse,
)
@management_endpoint_wrapper
async def get_user_daily_activity(
    start_date: str | None = fastapi.Query(
        default=None,
        description="Start date in YYYY-MM-DD format",
    ),
    end_date: str | None = fastapi.Query(
        default=None,
        description="End date in YYYY-MM-DD format",
    ),
    model: str | None = fastapi.Query(
        default=None,
        description="Filter by specific model",
    ),
    api_key: str | None = fastapi.Query(
        default=None,
        description="Filter by specific API key",
    ),
    user_id: str | None = fastapi.Query(
        default=None,
        description="Filter by specific user ID. Admins can filter by any user or omit for global view. Non-admins must provide their own user_id.",
    ),
    page: int = fastapi.Query(default=1, description="Page number for pagination", ge=1),
    page_size: int = fastapi.Query(default=50, description="Items per page", ge=1, le=1000),
    timezone: int | None = fastapi.Query(
        default=None,
        description="Timezone offset in minutes from UTC (e.g., 480 for PST). "
        "Matches JavaScript's Date.getTimezoneOffset() convention.",
    ),
    include_current_utc_day: bool = fastapi.Query(
        default=False,
        description="When the range ends on the caller's current local day, extend it to "
        "today's UTC bucket so spend written after the caller's local midnight (in UTC "
        "terms) is included. Requires the timezone parameter. Historical ranges are "
        "never extended.",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SpendAnalyticsPaginatedResponse:
    """
    [BETA] This is a beta endpoint. It will change.

    Meant to optimize querying spend data for analytics for a user.

    Returns:
    (by date)
    - spend
    - prompt_tokens
    - completion_tokens
    - cache_read_input_tokens
    - cache_creation_input_tokens
    - total_tokens
    - api_requests
    - breakdown by model, api_key, provider
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    try:
        is_admin: Final = _user_has_admin_view(user_api_key_dict)

        if is_admin:
            entity_id = user_id  # None means global view, otherwise filter by user
        else:
            caller_user_id: Final = require_caller_user_id_for_non_admin(user_api_key_dict)
            if user_id is None:
                user_id = caller_user_id
            if user_id != caller_user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": "Non-admin users can only view their own spend data."},
                )
            entity_id = user_id

        return await get_daily_activity(
            prisma_client=prisma_client,
            table_name="litellm_dailyuserspend",
            entity_id_field="user_id",
            entity_id=entity_id,
            entity_metadata_field=None,
            start_date=start_date,
            end_date=end_date,
            model=model,
            api_key=api_key,
            page=page,
            page_size=page_size,
            timezone_offset_minutes=timezone,
            include_current_utc_day=include_current_utc_day,
            resolve_entity_metadata=lambda records: _resolve_user_email_metadata(prisma_client, records),
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("/spend/daily/analytics: Exception occured - %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {e}"},
        )


@router.get(
    "/user/daily/activity/aggregated",
    tags=["Budget & Spend Tracking", "Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SpendAnalyticsPaginatedResponse,
)
@management_endpoint_wrapper
async def get_user_daily_activity_aggregated(
    start_date: str | None = fastapi.Query(
        default=None,
        description="Start date in YYYY-MM-DD format",
    ),
    end_date: str | None = fastapi.Query(
        default=None,
        description="End date in YYYY-MM-DD format",
    ),
    model: str | None = fastapi.Query(
        default=None,
        description="Filter by specific model",
    ),
    api_key: str | None = fastapi.Query(
        default=None,
        description="Filter by specific API key",
    ),
    user_id: str | None = fastapi.Query(
        default=None,
        description="Filter by specific user ID. Admins can filter by any user or omit for global view. Non-admins must provide their own user_id.",
    ),
    timezone: int | None = fastapi.Query(
        default=None,
        description="Timezone offset in minutes from UTC (e.g., 480 for PST). "
        "Matches JavaScript's Date.getTimezoneOffset() convention.",
    ),
    include_current_utc_day: bool = fastapi.Query(
        default=False,
        description="When the range ends on the caller's current local day, extend it to "
        "today's UTC bucket so spend written after the caller's local midnight (in UTC "
        "terms) is included. Requires the timezone parameter. Historical ranges are "
        "never extended.",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SpendAnalyticsPaginatedResponse:
    """
    Aggregated analytics for a user's daily activity without pagination.
    Returns the same response shape as the paginated endpoint with page metadata set to single-page.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    try:
        is_admin: Final = _user_has_admin_view(user_api_key_dict)

        if is_admin:
            entity_id = user_id  # None means global view, otherwise filter by user
        else:
            caller_user_id: Final = require_caller_user_id_for_non_admin(user_api_key_dict)
            if user_id is None:
                user_id = caller_user_id
            if user_id != caller_user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": "Non-admin users can only view their own spend data."},
                )
            entity_id = user_id

        return await get_daily_activity_aggregated(
            prisma_client=prisma_client,
            table_name="litellm_dailyuserspend",
            entity_id_field="user_id",
            entity_id=entity_id,
            entity_metadata_field=None,
            start_date=start_date,
            end_date=end_date,
            model=model,
            api_key=api_key,
            timezone_offset_minutes=timezone,
            include_current_utc_day=include_current_utc_day,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("/user/daily/activity/aggregated: Exception occured - %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {e}"},
        )
