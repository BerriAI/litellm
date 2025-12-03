"""
Internal User Management Endpoints


These are members of a Team on LiteLLM

/user/new
/user/update
/user/delete
/user/info
/user/list
"""

import asyncio
import json
import traceback
from litellm._uuid import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, cast

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks
from litellm.proxy.management_endpoints.common_daily_activity import (
    get_daily_activity,
    get_daily_activity_aggregated,
)
from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
from litellm.proxy.management_endpoints.key_management_endpoints import (
    generate_key_helper_fn,
    prepare_metadata_fields,
)
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.proxy.utils import handle_exception_on_proxy
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
    BulkUpdateUserRequest,
    BulkUpdateUserResponse,
    UserListResponse,
    UserUpdateResult,
)

if TYPE_CHECKING:
    from litellm.proxy.proxy_server import PrismaClient

router = APIRouter()


def _update_internal_new_user_params(data_json: dict, data: NewUserRequest) -> dict:
    if "user_id" in data_json and data_json["user_id"] is None:
        data_json["user_id"] = str(uuid.uuid4())

    auto_create_key = data_json.pop("auto_create_key", True)

    if auto_create_key is False:
        data_json["table_name"] = (
            "user"  # only create a user, don't create key if 'auto_create_key' set to False
        )

    if litellm.default_internal_user_params and (
        data.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and data.user_role != LitellmUserRoles.PROXY_ADMIN
    ):
        for key, value in litellm.default_internal_user_params.items():
            if key == "available_teams":
                continue
            elif key not in data_json or data_json[key] is None:
                data_json[key] = value
            elif (
                key == "models"
                and isinstance(data_json[key], list)
                and len(data_json[key]) == 0
            ):
                data_json[key] = value

    ## INTERNAL USER ROLE ONLY DEFAULT PARAMS ##
    if (
        data.user_role is not None
        and data.user_role == LitellmUserRoles.INTERNAL_USER.value
    ):
        if (
            litellm.max_internal_user_budget is not None
            and data_json.get("max_budget") is None
        ):
            data_json["max_budget"] = litellm.max_internal_user_budget

        if (
            litellm.internal_user_budget_duration is not None
            and data_json.get("budget_duration") is None
        ):
            data_json["budget_duration"] = litellm.internal_user_budget_duration

    data_json.pop("teams", None)  # handled separately
    return data_json


async def _check_duplicate_user_email(
    user_email: Optional[str], prisma_client: Any
) -> None:
    """
    Helper function to check if a user email already exists in the database.

    Args:
        user_email (Optional[str]): Email to check
        prisma_client (Any): Database client instance

    Raises:
        Exception: If database is not connected
        HTTPException: If user with email already exists
    """
    if user_email:
        if prisma_client is None:
            raise Exception("Database not connected")

        existing_user = await prisma_client.db.litellm_usertable.find_first(
            where={"user_email": {"equals": user_email.strip(), "mode": "insensitive"}}
        )

        if existing_user is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"User with email {existing_user.user_email} already exists"
                },
            )


async def _add_user_to_organizations(
    user_id: str,
    organizations: List[str],
    prisma_client: "PrismaClient",
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Add a user to organizations
    """
    from litellm.proxy.management_endpoints.organization_endpoints import (
        organization_member_add,
    )

    tasks = []
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
    user_email: Optional[str] = None,
    max_budget_in_team: Optional[float] = None,
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
        if e.status_code == 400 and (
            "already exists" in str(e) or "doesn't exist" in str(e)
        ):
            verbose_proxy_logger.debug(
                "litellm.proxy.management_endpoints.internal_user_endpoints.new_user(): User already exists in team - {}".format(
                    str(e)
                )
            )
        else:
            verbose_proxy_logger.debug(
                "litellm.proxy.management_endpoints.internal_user_endpoints.new_user(): Exception occured - {}".format(
                    str(e)
                )
            )
    except Exception as e:
        if "already exists" in str(e) or "doesn't exist" in str(e):
            verbose_proxy_logger.debug(
                "litellm.proxy.management_endpoints.internal_user_endpoints.new_user(): User already exists in team - {}".format(
                    str(e)
                )
            )
        elif (
            isinstance(e, ProxyException)
            and ProxyErrorTypes.team_member_already_in_team in e.type
        ):
            verbose_proxy_logger.debug(
                "litellm.proxy.management_endpoints.internal_user_endpoints.new_user(): User already exists in team - {}".format(
                    str(e)
                )
            )
        else:
            raise e


def check_if_default_team_set() -> Optional[Union[List[str], List[NewUserRequestTeam]]]:
    if litellm.default_internal_user_params is None:
        return None
    teams = litellm.default_internal_user_params.get("teams")
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
    user_email: Optional[str],
    user_api_key_dict: UserAPIKeyAuth,
    teams: Union[List[str], List[NewUserRequestTeam]],
    prisma_client: "PrismaClient",
):
    tasks = []
    for team in teams:
        user_role: Literal["user", "admin"] = "user"
        if isinstance(team, str):
            team_id = team
        elif isinstance(team, NewUserRequestTeam):
            team_id = team.team_id
            user_role = team.user_role
        else:
            raise ValueError(f"Invalid team type: {type(team)}")

        tasks.append(
            _add_user_to_team(
                user_id=user_id,
                team_id=team_id,
                user_email=user_email,
                user_api_key_dict=user_api_key_dict,
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
    - permissions: Optional[dict] - [Not Implemented Yet] User-specific permissions, eg. turning off pii masking.
    - metadata: Optional[dict] - Metadata for user, store information for user. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
    - max_parallel_requests: Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
    - soft_budget: Optional[float] - Get alerts when user crosses given budget, doesn't block requests.
    - model_max_budget: Optional[dict] - Model-specific max budget for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-budgets-to-keys)
    - model_rpm_limit: Optional[float] - Model-specific rpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)
    - model_tpm_limit: Optional[float] - Model-specific tpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)
    - spend: Optional[float] - Amount spent by user. Default is 0. Will be updated by proxy whenever user is used. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"), months ("1mo").
    - team_id: Optional[str] - [DEPRECATED PARAM] The team id of the user. Default is None. 
    - duration: Optional[str] - Duration for the key auto-created on `/user/new`. Default is None.
    - key_alias: Optional[str] - Alias for the key auto-created on `/user/new`. Default is None.
    - sso_user_id: Optional[str] - The id of the user in the SSO provider.
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - internal user-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"]}. IF null or {} then no object permission.
    - prompts: Optional[List[str]] - List of allowed prompts for the user. If specified, the user will only be able to use these specific prompts.
    - organizations: List[str] - List of organization id's the user is a member of
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
            raise HTTPException(
                status_code=400, detail=CommonProxyErrors.db_not_connected_error.value
            )

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail=CommonProxyErrors.db_not_connected_error.value,
            )
        # Check for duplicate email
        await _check_duplicate_user_email(data.user_email, prisma_client)

        # Check if license is over limit
        total_users = await prisma_client.db.litellm_usertable.count()
        if total_users and _license_check.is_over_limit(total_users=total_users):
            raise HTTPException(
                status_code=403,
                detail="License is over limit. Please contact support@berri.ai to upgrade your license.",
            )

        data_json = data.json()  # type: ignore
        data_json = _update_internal_new_user_params(data_json, data)
        teams = data.teams
        if teams is None:
            teams = check_if_default_team_set()
        organization_ids = cast(
            Optional[List[str]], data_json.pop("organizations", None)
        )

        response = await generate_key_helper_fn(request_type="user", **data_json)
        # Admin UI Logic
        # Add User to Team and Organization
        # if team_id passed add this user to the team
        _team_id = data_json.get("team_id", None)
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

        user_id = cast(Optional[str], response.get("user_id", None))

        if organization_ids is not None and user_id is not None:
            await _add_user_to_organizations(
                user_id=user_id,
                organizations=organization_ids,
                prisma_client=prisma_client,
                user_api_key_dict=user_api_key_dict,
            )

        special_keys = ["token", "token_id"]
        response_dict = {}
        for key, value in response.items():
            if key in NewUserResponse.model_fields.keys() and key not in special_keys:
                response_dict[key] = value

        response_dict["key"] = response.get("token", "")

        new_user_response = NewUserResponse(**response_dict)

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
        verbose_proxy_logger.exception(
            "/user/new: Exception occured - {}".format(str(e))
        )
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

    _data_to_return = {}
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
    team_list: Optional[Union[List[LiteLLM_TeamTable], List[TeamListResponseObject]]],
    team_id: str,
) -> Optional[Union[LiteLLM_TeamTable, LiteLLM_TeamMembership]]:
    if team_list is None:
        return None

    for team in team_list:
        if team.team_id == team_id:
            return team
    return None


def get_user_id_from_request(request: Request) -> Optional[str]:
    """
    Get the user id from the request
    """
    # Get the raw query string and parse it properly to handle + characters
    user_id: Optional[str] = None
    query_string = str(request.url.query)
    if "user_id=" in query_string:
        # Extract the user_id value from the raw query string
        import re
        from urllib.parse import unquote

        match = re.search(r"user_id=([^&]*)", query_string)
        if match:
            # Use unquote instead of unquote_plus to preserve + characters
            raw_user_id = unquote(match.group(1))
            user_id = raw_user_id
    return user_id


@router.get(
    "/user/info",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    # response_model=UserInfoResponse,
)
@management_endpoint_wrapper
async def user_info(
    request: Request,
    user_id: Optional[str] = fastapi.Query(
        default=None, description="User ID in the request parameters"
    ),
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
    from litellm.proxy.proxy_server import prisma_client

    try:
        # Handle URL encoding properly by getting user_id from the original request
        if (
            user_id is not None and " " in user_id
        ):  # if user_id is not None and contains a space, get the user_id from the request - this is to handle the case where the user_id is encoded in the url
            user_id = get_user_id_from_request(request=request)

        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        if (
            user_id is None
            and user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            return await _get_user_info_for_proxy_admin()
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

        ## GET ALL TEAMS ##
        team_list = []
        team_id_list = []
        from litellm.proxy.management_endpoints.team_endpoints import list_team

        teams_1 = await list_team(
            http_request=Request(
                scope={"type": "http", "path": "/user/info"},
            ),
            user_id=user_id,
            user_api_key_dict=user_api_key_dict,
        )

        if teams_1 is not None and isinstance(teams_1, list):
            team_list = teams_1
            for team in teams_1:
                team_id_list.append(team.team_id)

        teams_2: Optional[Any] = None
        if user_info is not None:
            # *NEW* get all teams in user 'teams' field
            teams_2 = await prisma_client.get_data(
                team_id_list=user_info.teams, table_name="team", query_type="find_all"
            )

            if teams_2 is not None and isinstance(teams_2, list):
                for team in teams_2:
                    if team.team_id not in team_id_list:
                        team_list.append(team)
                        team_id_list.append(team.team_id)

        elif (
            user_api_key_dict.user_id is not None and user_id is None
        ):  # the key querying the endpoint is the one asking for it's teams
            caller_user_info = await prisma_client.get_data(
                user_id=user_api_key_dict.user_id
            )
            # *NEW* get all teams in user 'teams' field
            if caller_user_info is not None:
                teams_2 = await prisma_client.get_data(
                    team_id_list=caller_user_info.teams,
                    table_name="team",
                    query_type="find_all",
                )

            if teams_2 is not None and isinstance(teams_2, list):
                for team in teams_2:
                    if team.team_id not in team_id_list:
                        team_list.append(team)
                        team_id_list.append(team.team_id)

        ## GET ALL KEYS ##
        keys = await prisma_client.get_data(
            user_id=user_id,
            table_name="key",
            query_type="find_all",
        )

        if user_info is None and keys is not None:
            ## make sure we still return a total spend ##
            spend = 0
            for k in keys:
                spend += getattr(k, "spend", 0)
            user_info = {"spend": spend}

        ## REMOVE HASHED TOKEN INFO before returning ##
        returned_keys = _process_keys_for_user_info(keys=keys, all_teams=teams_1)
        team_list.sort(key=lambda x: (getattr(x, "team_alias", "") or ""))
        _user_info = (
            user_info.model_dump() if isinstance(user_info, BaseModel) else user_info
        )
        response_data = UserInfoResponse(
            user_id=user_id, user_info=_user_info, keys=returned_keys, teams=team_list
        )

        return response_data
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.user_info(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


async def _get_user_info_for_proxy_admin():
    """
    Admin UI Endpoint - Returns All Teams and Keys when Proxy Admin is querying

    - get all teams in LiteLLM_TeamTable
    - get all keys in LiteLLM_VerificationToken table

    Why separate helper for proxy admin ?
        - To get Faster UI load times, get all teams and virtual keys in 1 query
    """

    from litellm.proxy.proxy_server import prisma_client

    sql_query = """
        SELECT 
            (SELECT json_agg(t.*) FROM "LiteLLM_TeamTable" t) as teams,
            (SELECT json_agg(k.*) FROM "LiteLLM_VerificationToken" k WHERE k.team_id != 'litellm-dashboard' OR k.team_id IS NULL) as keys
    """
    if prisma_client is None:
        raise Exception(
            "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
        )

    results = await prisma_client.db.query_raw(sql_query)

    verbose_proxy_logger.debug("results_keys: %s", results)

    _keys_in_db: List = results[0]["keys"] or []
    # cast all keys to LiteLLM_VerificationToken
    keys_in_db = []
    for key in _keys_in_db:
        if key.get("models") is None:
            key["models"] = []
        keys_in_db.append(LiteLLM_VerificationToken(**key))

    # cast all teams to LiteLLM_TeamTable
    _teams_in_db: List = results[0]["teams"] or []
    _teams_in_db = [LiteLLM_TeamTable(**team) for team in _teams_in_db]
    _teams_in_db.sort(key=lambda x: (getattr(x, "team_alias", "") or ""))
    returned_keys = _process_keys_for_user_info(keys=keys_in_db, all_teams=_teams_in_db)
    return UserInfoResponse(
        user_id=None,
        user_info=None,
        keys=returned_keys,
        teams=_teams_in_db,
    )


def _process_keys_for_user_info(
    keys: Optional[List[LiteLLM_VerificationToken]],
    all_teams: Optional[Union[List[LiteLLM_TeamTable], List[TeamListResponseObject]]],
):
    from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
    from litellm.proxy.proxy_server import general_settings, litellm_master_key_hash

    returned_keys = []
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
                _key: dict = key.model_dump()  # noqa
            except Exception:
                # if using pydantic v1
                _key = key.dict()
            
            # Filter out UI session tokens (team_id="litellm-dashboard")
            if _key.get("team_id") == UI_SESSION_TOKEN_TEAM_ID:
                continue
            
            if (
                "team_id" in _key
                and _key["team_id"] is not None
                and _key["team_id"] != "litellm-dashboard"
            ):
                team_info = get_team_from_list(
                    team_list=all_teams, team_id=_key["team_id"]
                )
                if team_info is not None:
                    team_alias = getattr(team_info, "team_alias", None)
                    _key["team_alias"] = team_alias
                else:
                    _key["team_alias"] = None
            else:
                _key["team_alias"] = "None"
            returned_keys.append(_key)
    return returned_keys


def _update_internal_user_params(
    data_json: dict, data: Union[UpdateUserRequest, UpdateUserRequestNoUserIDorEmail]
) -> dict:
    non_default_values = {}
    for k, v in data_json.items():
        if (
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

        non_default_values["budget_reset_at"] = get_budget_reset_time(
            budget_duration=non_default_values["budget_duration"]
        )

    if "max_budget" not in non_default_values:
        if (
            is_internal_user and litellm.max_internal_user_budget is not None
        ):  # applies internal user limits, if user role updated
            non_default_values["max_budget"] = litellm.max_internal_user_budget

    if (
        "budget_duration" not in non_default_values
    ):  # applies internal user limits, if user role updated
        if is_internal_user and litellm.internal_user_budget_duration is not None:
            non_default_values["budget_duration"] = (
                litellm.internal_user_budget_duration
            )
            from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

            non_default_values["budget_reset_at"] = get_budget_reset_time(
                budget_duration=non_default_values["budget_duration"]
            )

    return non_default_values


async def _update_single_user_helper(
    user_request: UpdateUserRequest,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Helper function to update a single user.
    Used by both user_update and bulk_user_update endpoints.

    Returns the updated user data or raises an exception on failure.
    """
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise Exception("Not connected to DB!")

    # Validate user identifier
    if not user_request.user_id and not user_request.user_email:
        raise ValueError("Either user_id or user_email must be provided")

    # Convert to data format expected by update logic
    data_json: dict = user_request.model_dump(exclude_unset=True)

    # Apply update transformations (reuse existing logic)
    non_default_values = _update_internal_user_params(
        data_json=data_json, data=user_request
    )

    # Get existing user data for audit logging and metadata preparation
    existing_user_row: Optional[BaseModel] = None
    if user_request.user_id:
        existing_user_row = await prisma_client.db.litellm_usertable.find_first(
            where={"user_id": user_request.user_id}
        )
    elif user_request.user_email:
        existing_user_row = await prisma_client.db.litellm_usertable.find_first(
            where={"user_email": user_request.user_email}
        )

    if existing_user_row is not None:
        existing_user_row = LiteLLM_UserTable(
            **existing_user_row.model_dump(exclude_none=True)
        )
        if not can_user_call_user_update(
            user_api_key_dict=user_api_key_dict,
            user_info=existing_user_row,
        ):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "User does not have permission to update this user. Only PROXY_ADMIN can update other users."
                },
            )

    existing_metadata = (
        cast(Dict, getattr(existing_user_row, "metadata", {}) or {})
        if existing_user_row is not None
        else {}
    )

    non_default_values = prepare_metadata_fields(
        data=user_request,
        non_default_values=non_default_values,
        existing_metadata=existing_metadata or {},
    )

    # Perform the update
    response: Optional[Dict[str, Any]] = None

    if user_request.user_id and len(user_request.user_id) > 0:
        non_default_values["user_id"] = user_request.user_id
        response = await prisma_client.update_data(
            user_id=user_request.user_id,
            data=non_default_values,
            table_name="user",
        )
    elif user_request.user_email:
        # Handle email-based updates
        existing_user_rows = await prisma_client.get_data(
            key_val={"user_email": user_request.user_email},
            table_name="user",
            query_type="find_all",
        )

        if (
            existing_user_rows
            and isinstance(existing_user_rows, list)
            and len(existing_user_rows) > 0
        ):
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
            response = await prisma_client.insert_data(
                data=non_default_values, table_name="user"
            )

    # Create audit log for successful update
    if response is not None:
        try:
            updated_user_row = await prisma_client.db.litellm_usertable.find_first(
                where={"user_id": response["user_id"]}
            )

            if updated_user_row:
                user_row_typed = LiteLLM_UserTable(
                    **updated_user_row.model_dump(exclude_none=True)
                )

                # Create audit log asynchronously
                asyncio.create_task(
                    UserManagementEventHooks.create_internal_user_audit_log(
                        user_id=user_row_typed.user_id,
                        action="updated",
                        litellm_changed_by=litellm_changed_by
                        or user_api_key_dict.user_id,
                        user_api_key_dict=user_api_key_dict,
                        litellm_proxy_admin_name=litellm_proxy_admin_name,
                        before_value=(
                            existing_user_row.model_dump_json(exclude_none=True)
                            if existing_user_row
                            else None
                        ),
                        after_value=user_row_typed.model_dump_json(exclude_none=True),
                    )
                )
        except Exception as audit_error:
            verbose_proxy_logger.warning(
                f"Failed to create audit log for user {response.get('user_id')}: {audit_error}"
            )

    if response is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Failed to update user"},
        )
    return response


def can_user_call_user_update(
    user_api_key_dict: UserAPIKeyAuth,
    user_info: LiteLLM_UserTable,
) -> bool:
    """
    Helper to check if the user has access to the key's info
    """
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return True
    elif user_api_key_dict.user_id == user_info.user_id:
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
        - permissions: Optional[dict] - [Not Implemented Yet] User-specific permissions, eg. turning off pii masking.
        - metadata: Optional[dict] - Metadata for user, store information for user. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
        - max_parallel_requests: Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
        - soft_budget: Optional[float] - Get alerts when user crosses given budget, doesn't block requests.
        - model_max_budget: Optional[dict] - Model-specific max budget for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-budgets-to-keys)
        - model_rpm_limit: Optional[float] - Model-specific rpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)
        - model_tpm_limit: Optional[float] - Model-specific tpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)
        - spend: Optional[float] - Amount spent by user. Default is 0. Will be updated by proxy whenever user is used. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"), months ("1mo").
        - team_id: Optional[str] - [DEPRECATED PARAM] The team id of the user. Default is None. 
        - duration: Optional[str] - [NOT IMPLEMENTED].
        - key_alias: Optional[str] - [NOT IMPLEMENTED].
        - object_permission: Optional[LiteLLM_ObjectPermissionBase] - internal user-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"]}. IF null or {} then no object permission.
        - prompts: Optional[List[str]] - List of allowed prompts for the user. If specified, the user will only be able to use these specific prompts.
    
    """
    try:
        verbose_proxy_logger.debug("/user/update: Received data = %s", data)

        response = await _update_single_user_helper(
            user_request=data,
            user_api_key_dict=user_api_key_dict,
        )
        return response
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.user_update(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
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
    users_to_update: List[UpdateUserRequest],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str] = None,
) -> BulkUpdateUserResponse:
    results: List[UserUpdateResult] = []
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
                        user_id=(
                            response.get("user_id")
                            if response
                            else user_request.user_id
                        ),
                        user_email=user_request.user_email,
                        success=True,
                        updated_user=response,
                    )
                )
                successful_updates += 1
            except Exception as e:
                verbose_proxy_logger.exception(
                    f"Failed to update user {user_request.user_id or user_request.user_email}: {e}"
                )
                # Record failure
                error_message = str(e)
                verbose_proxy_logger.error(
                    f"Failed to update user {user_request.user_id or user_request.user_email}: {error_message}"
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
        verbose_proxy_logger.exception(f"Failed to update users: {e}")
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
    litellm_changed_by: Optional[str] = Header(
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

    # Determine the list of users to update
    users_to_update: Union[
        List[UpdateUserRequest], List[UpdateUserRequestNoUserIDorEmail]
    ] = []

    if data.all_users and data.user_updates:
        # Optimized path for updating all users directly in database
        all_users_in_db = await prisma_client.db.litellm_usertable.find_many(
            order={"created_at": "desc"}
        )

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
        data_json: dict = data.user_updates.model_dump(exclude_unset=True)
        non_default_values = _update_internal_user_params(
            data_json=data_json, data=data.user_updates
        )

        # Remove user identification fields since we're updating by user_id
        non_default_values.pop("user_id", None)
        non_default_values.pop("user_email", None)

        successful_updates = 0
        failed_updates = 0
        results: List[UserUpdateResult] = []

        try:
            # Perform bulk database update
            await prisma_client.db.litellm_usertable.update_many(
                where={}, data=non_default_values  # Update all users
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
                        litellm_changed_by=litellm_changed_by
                        or user_api_key_dict.user_id,
                        user_api_key_dict=user_api_key_dict,
                        litellm_proxy_admin_name=litellm_proxy_admin_name,
                        before_value=f"Updated {len(all_users_in_db)} users",
                        after_value=json.dumps(non_default_values),
                    )
                )
            except Exception as audit_error:
                verbose_proxy_logger.warning(
                    f"Failed to create bulk audit log: {audit_error}"
                )

        except Exception as e:
            verbose_proxy_logger.exception(f"Failed to perform bulk update: {e}")
            # Fall back to individual updates if bulk update fails
            for user in all_users_in_db:
                user_update_request = data.user_updates.model_copy()
                user_update_request.user_id = user.user_id
                users_to_update.append(user_update_request)  # type: ignore

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
        users_to_update=cast(List[UpdateUserRequest], users_to_update),
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=litellm_changed_by,
    )


async def get_user_key_counts(
    prisma_client,
    user_ids: Optional[List[str]] = None,
):
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

    result = {}

    # Get count for each user_id individually
    for user_id in user_ids:
        count = await prisma_client.db.litellm_verificationtoken.count(
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


def _validate_sort_params(
    sort_by: Optional[str], sort_order: str
) -> Optional[Dict[str, str]]:
    order_by: Dict[str, str] = {}

    if sort_by is None:
        return None
    # Validate sort_by is a valid column
    valid_columns = [
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
            detail={
                "error": f"Invalid sort column. Must be one of: {', '.join(valid_columns)}"
            },
        )

    # Validate sort_order
    if sort_order.lower() not in ["asc", "desc"]:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid sort order. Must be 'asc' or 'desc'"},
        )

    order_by[sort_by] = sort_order.lower()

    return order_by


@router.get(
    "/user/list",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UserListResponse,
)
async def get_users(
    role: Optional[str] = fastapi.Query(
        default=None, description="Filter users by role"
    ),
    user_ids: Optional[str] = fastapi.Query(
        default=None, description="Get list of users by user_ids"
    ),
    sso_user_ids: Optional[str] = fastapi.Query(
        default=None, description="Get list of users by sso_user_id"
    ),
    user_email: Optional[str] = fastapi.Query(
        default=None, description="Filter users by partial email match"
    ),
    team: Optional[str] = fastapi.Query(
        default=None, description="Filter users by team id"
    ),
    page: int = fastapi.Query(default=1, ge=1, description="Page number"),
    page_size: int = fastapi.Query(
        default=25, ge=1, le=100, description="Number of items per page"
    ),
    sort_by: Optional[str] = fastapi.Query(
        default=None,
        description="Column to sort by (e.g. 'user_id', 'user_email', 'created_at', 'spend')",
    ),
    sort_order: str = fastapi.Query(
        default="asc", description="Sort order ('asc' or 'desc')"
    ),
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
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": f"No db connected. prisma client={prisma_client}"},
        )

    # Calculate skip and take for pagination
    skip = (page - 1) * page_size

    # Build where conditions based on provided parameters
    where_conditions: Dict[str, Any] = {}

    if role:
        where_conditions["user_role"] = role

    if user_ids and isinstance(user_ids, str):
        user_id_list = [uid.strip() for uid in user_ids.split(",") if uid.strip()]
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
        sso_id_list = [sid.strip() for sid in sso_user_ids.split(",") if sid.strip()]
        where_conditions["sso_user_id"] = {
            "in": sso_id_list,
        }

    ## Filter any none fastapi.Query params - e.g. where_conditions: {'user_email': {'contains': Query(None), 'mode': 'insensitive'}, 'teams': {'has': Query(None)}}
    where_conditions = {k: v for k, v in where_conditions.items() if v is not None}

    # Build order_by conditions

    order_by: Optional[Dict[str, str]] = (
        _validate_sort_params(sort_by, sort_order)
        if sort_by is not None and isinstance(sort_by, str)
        else None
    )

    users = await prisma_client.db.litellm_usertable.find_many(
        where=where_conditions,
        skip=skip,
        take=page_size,
        order=(
            order_by if order_by else {"created_at": "desc"}
        ),  # Default to created_at desc if no sort specified
    )

    # Get total count of user rows
    total_count = await prisma_client.db.litellm_usertable.count(where=where_conditions)

    # Get key count for each user
    if users is not None:
        user_key_counts = await get_user_key_counts(
            prisma_client, [user.user_id for user in users]
        )
    else:
        user_key_counts = {}

    verbose_proxy_logger.debug(f"Total count of users: {total_count}")

    # Calculate total pages
    total_pages = -(-total_count // page_size)  # Ceiling division

    # Prepare response
    user_list: List[LiteLLM_UserTableWithKeyCount] = []
    if users is not None:
        for user in users:
            user_list.append(
                LiteLLM_UserTableWithKeyCount(
                    **user.model_dump(), key_count=user_key_counts.get(user.user_id, 0)
                )
            )
    else:
        user_list = []

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
    litellm_changed_by: Optional[str] = Header(
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
    from litellm.proxy.proxy_server import (
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.user_ids is None:
        raise HTTPException(status_code=400, detail={"error": "No user id passed in"})

    # check that all teams passed exist
    for user_id in data.user_ids:
        user_row = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_id}
        )

        if user_row is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"User not found, passed user_id={user_id}"},
            )
        else:
            # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
            # we do this after the first for loop, since first for loop is for validation. we only want this inserted after validation passes
            if litellm.store_audit_logs is True:
                # make an audit log for each team deleted
                _user_row = user_row.json(exclude_none=True)

                asyncio.create_task(
                    create_audit_log_for_update(
                        request_data=LiteLLM_AuditLogs(
                            id=str(uuid.uuid4()),
                            updated_at=datetime.now(timezone.utc),
                            changed_by=litellm_changed_by
                            or user_api_key_dict.user_id
                            or litellm_proxy_admin_name,
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
        fetch_all_teams = await prisma_client.db.litellm_teamtable.find_many(
            where={"team_id": {"in": user_row.teams}}
        )
        teams_to_update = []
        for team in fetch_all_teams:
            is_member_in_team, new_team_members = _cleanup_members_with_roles(
                existing_team_row=LiteLLM_TeamTable(**team.model_dump()),
                data=TeamMemberDeleteRequest(
                    team_id=team.team_id,
                    user_id=user_row.user_id,
                    user_email=user_row.user_email,
                ),
            )
            if is_member_in_team:
                _db_new_team_members: List[dict] = [
                    m.model_dump() for m in new_team_members
                ]
                team.members_with_roles = json.dumps(_db_new_team_members)
                teams_to_update.append(team)

        ## update teams

        for team in teams_to_update:
            await prisma_client.db.litellm_teamtable.update(
                where={"team_id": team.team_id},
                data={"members_with_roles": team.members_with_roles},
            )
    # End of Audit logging

    ## DELETE ASSOCIATED KEYS
    await prisma_client.db.litellm_verificationtoken.delete_many(
        where={"user_id": {"in": data.user_ids}}
    )

    ## DELETE ASSOCIATED INVITATION LINKS
    await prisma_client.db.litellm_invitationlink.delete_many(
        where={"user_id": {"in": data.user_ids}}
    )

    ## DELETE ASSOCIATED ORGANIZATION MEMBERSHIPS
    await prisma_client.db.litellm_organizationmembership.delete_many(
        where={"user_id": {"in": data.user_ids}}
    )

    ## DELETE ASSOCIATED TEAM MEMBERSHIPS
    await prisma_client.db.litellm_teammembership.delete_many(
        where={"user_id": {"in": data.user_ids}}
    )

    ## DELETE USERS
    deleted_users = await prisma_client.db.litellm_usertable.delete_many(
        where={"user_id": {"in": data.user_ids}}
    )

    return deleted_users


async def add_internal_user_to_organization(
    user_id: str,
    organization_id: str,
    user_role: LitellmUserRoles,
):
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
        organization_row = await prisma_client.db.litellm_organizationtable.find_unique(
            where={"organization_id": organization_id}
        )
        if organization_row is None:
            raise Exception(
                f"Organization not found, passed organization_id={organization_id}"
            )

        # Create a new organization membership entry
        new_membership = await prisma_client.db.litellm_organizationmembership.create(
            data={
                "user_id": user_id,
                "organization_id": organization_id,
                "user_role": user_role,
                # Note: You can also set budget within an organization if needed
            }
        )

        return new_membership
    except Exception as e:
        raise Exception(f"Failed to add user to organization: {str(e)}")


@router.get(
    "/user/filter/ui",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
    responses={
        200: {"model": List[LiteLLM_UserTableFiltered]},
    },
)
async def ui_view_users(
    user_id: Optional[str] = fastapi.Query(
        default=None, description="User ID in the request parameters"
    ),
    user_email: Optional[str] = fastapi.Query(
        default=None, description="User email in the request parameters"
    ),
    page: int = fastapi.Query(
        default=1, description="Page number for pagination", ge=1
    ),
    page_size: int = fastapi.Query(
        default=50, description="Number of items per page", ge=1, le=100
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [PROXY-ADMIN ONLY]Filter users based on partial match of user_id or email with pagination.

    Args:
        user_id (Optional[str]): Partial user ID to search for
        user_email (Optional[str]): Partial email to search for
        page (int): Page number for pagination (starts at 1)
        page_size (int): Number of items per page (max 100)
        user_api_key_dict (UserAPIKeyAuth): User authentication information

    Returns:
        List[LiteLLM_SpendLogs]: Paginated list of matching user records
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    try:
        # Calculate offset for pagination
        skip = (page - 1) * page_size

        # Build where conditions based on provided parameters
        where_conditions = {}

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

        # Query users with pagination and filters
        users: Optional[List[BaseModel]] = (
            await prisma_client.db.litellm_usertable.find_many(
                where=where_conditions,
                skip=skip,
                take=page_size,
                order={"created_at": "desc"},
            )
        )

        if not users:
            return []

        return [LiteLLM_UserTableFiltered(**user.model_dump()) for user in users]

    except Exception as e:
        verbose_proxy_logger.exception(f"Error searching users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching users: {str(e)}")


# Using shared metric helper implementations from common_daily_activity


@router.get(
    "/user/daily/activity",
    tags=["Budget & Spend Tracking", "Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SpendAnalyticsPaginatedResponse,
)
@management_endpoint_wrapper
async def get_user_daily_activity(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Start date in YYYY-MM-DD format",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="End date in YYYY-MM-DD format",
    ),
    model: Optional[str] = fastapi.Query(
        default=None,
        description="Filter by specific model",
    ),
    api_key: Optional[str] = fastapi.Query(
        default=None,
        description="Filter by specific API key",
    ),
    page: int = fastapi.Query(
        default=1, description="Page number for pagination", ge=1
    ),
    page_size: int = fastapi.Query(
        default=50, description="Items per page", ge=1, le=1000
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
        entity_id: Optional[str] = None
        if not _user_has_admin_view(user_api_key_dict):
            entity_id = user_api_key_dict.user_id

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
        )

    except Exception as e:
        verbose_proxy_logger.exception(
            "/spend/daily/analytics: Exception occured - {}".format(str(e))
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {str(e)}"},
        )


@router.get(
    "/user/daily/activity/aggregated",
    tags=["Budget & Spend Tracking", "Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SpendAnalyticsPaginatedResponse,
)
@management_endpoint_wrapper
async def get_user_daily_activity_aggregated(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Start date in YYYY-MM-DD format",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="End date in YYYY-MM-DD format",
    ),
    model: Optional[str] = fastapi.Query(
        default=None,
        description="Filter by specific model",
    ),
    api_key: Optional[str] = fastapi.Query(
        default=None,
        description="Filter by specific API key",
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
        entity_id: Optional[str] = None
        if not _user_has_admin_view(user_api_key_dict):
            entity_id = user_api_key_dict.user_id

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
        )

    except Exception as e:
        verbose_proxy_logger.exception(
            "/user/daily/activity/aggregated: Exception occured - {}".format(str(e))
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {str(e)}"},
        )
