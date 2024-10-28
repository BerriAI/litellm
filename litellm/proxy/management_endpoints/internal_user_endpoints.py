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
import copy
import json
import re
import secrets
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _duration_in_seconds,
    generate_key_helper_fn,
)
from litellm.proxy.management_helpers.utils import (
    add_new_member,
    management_endpoint_wrapper,
)

router = APIRouter()


def _update_internal_user_params(data_json: dict, data: NewUserRequest) -> dict:
    if "user_id" in data_json and data_json["user_id"] is None:
        data_json["user_id"] = str(uuid.uuid4())
    auto_create_key = data_json.pop("auto_create_key", True)
    if auto_create_key is False:
        data_json["table_name"] = (
            "user"  # only create a user, don't create key if 'auto_create_key' set to False
        )

    is_internal_user = False
    if data.user_role == LitellmUserRoles.INTERNAL_USER:
        is_internal_user = True
        if litellm.default_internal_user_params:
            for key, value in litellm.default_internal_user_params.items():
                if key not in data_json or data_json[key] is None:
                    data_json[key] = value
                elif (
                    key == "models"
                    and isinstance(data_json[key], list)
                    and len(data_json[key]) == 0
                ):
                    data_json[key] = value

    if "max_budget" in data_json and data_json["max_budget"] is None:
        if is_internal_user and litellm.max_internal_user_budget is not None:
            data_json["max_budget"] = litellm.max_internal_user_budget

    if "budget_duration" in data_json and data_json["budget_duration"] is None:
        if is_internal_user and litellm.internal_user_budget_duration is not None:
            data_json["budget_duration"] = litellm.internal_user_budget_duration

    return data_json


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
    - budget_duration: Optional[str] - Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
    - tpm_limit: Optional[int] - Specify tpm limit for a given user (Tokens per minute)
    - rpm_limit: Optional[int] - Specify rpm limit for a given user (Requests per minute)
    - auto_create_key: bool - Default=True. Flag used for returning a key as part of the /user/new response

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
    from litellm.proxy.proxy_server import general_settings, proxy_logging_obj

    data_json = data.json()  # type: ignore
    data_json = _update_internal_user_params(data_json, data)
    response = await generate_key_helper_fn(request_type="user", **data_json)

    # Admin UI Logic
    # Add User to Team and Organization
    # if team_id passed add this user to the team
    if data_json.get("team_id", None) is not None:
        from litellm.proxy.management_endpoints.team_endpoints import team_member_add

        await team_member_add(
            data=TeamMemberAddRequest(
                team_id=data_json.get("team_id", None),
                member=Member(
                    user_id=data_json.get("user_id", None),
                    role="user",
                    user_email=data_json.get("user_email", None),
                ),
            ),
            http_request=Request(
                scope={"type": "http", "path": "/user/new"},
            ),
            user_api_key_dict=user_api_key_dict,
        )

    if data.send_invite_email is True:
        # check if user has setup email alerting
        if "email" not in general_settings.get("alerting", []):
            raise ValueError(
                "Email alerting not setup on config.yaml. Please set `alerting=['email']. \nDocs: https://docs.litellm.ai/docs/proxy/email`"
            )

        event = WebhookEvent(
            event="internal_user_created",
            event_group="internal_user",
            event_message="Welcome to LiteLLM Proxy",
            token=response.get("token", ""),
            spend=response.get("spend", 0.0),
            max_budget=response.get("max_budget", 0.0),
            user_id=response.get("user_id", None),
            user_email=response.get("user_email", None),
            team_id=response.get("team_id", "Default Team"),
            key_alias=response.get("key_alias", None),
        )

        # If user configured email alerting - send an Email letting their end-user know the key was created
        asyncio.create_task(
            proxy_logging_obj.slack_alerting_instance.send_key_created_or_user_invited_email(
                webhook_event=event,
            )
        )

    return NewUserResponse(
        key=response.get("token", ""),
        expires=response.get("expires", None),
        max_budget=response["max_budget"],
        user_id=response["user_id"],
        user_role=response.get("user_role", None),
        user_email=response.get("user_email", None),
        user_alias=response.get("user_alias", None),
        teams=response.get("teams", None),
        team_id=response.get("team_id", None),
        metadata=response.get("metadata", None),
        models=response.get("models", None),
        tpm_limit=response.get("tpm_limit", None),
        rpm_limit=response.get("rpm_limit", None),
        budget_duration=response.get("budget_duration", None),
    )


@router.post(
    "/user/auth",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def user_auth(request: Request):
    """
    Allows UI ("https://dashboard.litellm.ai/", or self-hosted - os.getenv("LITELLM_HOSTED_UI")) to request a magic link to be sent to user email, for auth to proxy.

    Only allows emails from accepted email subdomains.

    Rate limit: 1 request every 60s.

    Only works, if you enable 'allow_user_auth' in general settings:
    e.g.:
    ```yaml
    general_settings:
        allow_user_auth: true
    ```

    Requirements:
    SMTP server details saved in .env:
    - os.environ["SMTP_HOST"]
    - os.environ["SMTP_PORT"]
    - os.environ["SMTP_USERNAME"]
    - os.environ["SMTP_PASSWORD"]
    - os.environ["SMTP_SENDER_EMAIL"]
    """
    from litellm.proxy.proxy_server import prisma_client
    from litellm.proxy.utils import send_email

    data = await request.json()  # type: ignore
    user_email = data["user_email"]
    page_params = data["page"]
    if user_email is None:
        raise HTTPException(status_code=400, detail="User email is none")

    if prisma_client is None:  # if no db connected, raise an error
        raise Exception("No connected db.")

    ### Check if user email in user table
    response = await prisma_client.get_generic_data(
        key="user_email", value=user_email, table_name="users"
    )
    ### if so - generate a 24 hr key with that user id
    if response is not None:
        user_id = response.user_id  # type: ignore
        response = await generate_key_helper_fn(
            request_type="key",
            **{"duration": "24hr", "models": [], "aliases": {}, "config": {}, "spend": 0, "user_id": user_id},  # type: ignore
        )
    else:  ### else - create new user
        response = await generate_key_helper_fn(
            request_type="key",
            **{"duration": "24hr", "models": [], "aliases": {}, "config": {}, "spend": 0, "user_email": user_email},  # type: ignore
        )

    base_url = os.getenv("LITELLM_HOSTED_UI", "https://dashboard.litellm.ai/")

    params = {
        "sender_name": "LiteLLM Proxy",
        "receiver_email": user_email,
        "subject": "Your Magic Link",
        "html": f"<strong> Follow this  link, to login:\n\n{base_url}user/?token={response['token']}&user_id={response['user_id']}&page={page_params}</strong>",
    }

    await send_email(**params)
    return "Email sent!"


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


@router.get(
    "/user/info",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    # response_model=UserInfoResponse,
)
@management_endpoint_wrapper
async def user_info(  # noqa: PLR0915
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
    curl -X GET 'http://localhost:8000/user/info?user_id=krrish7%40berri.ai' \
    --header 'Authorization: Bearer sk-1234'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        litellm_master_key_hash,
        prisma_client,
    )

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        ## GET USER ROW ##
        if user_id is not None:
            user_info = await prisma_client.get_data(user_id=user_id)
        else:
            user_info = None
        ## GET ALL TEAMS ##
        team_list = []
        team_id_list = []
        # get all teams user belongs to
        # teams_1 = await prisma_client.get_data(
        #     user_id=user_id, table_name="team", query_type="find_all"
        # )
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
            if (
                getattr(caller_user_info, "user_role", None)
                == LitellmUserRoles.PROXY_ADMIN
            ):
                from litellm.proxy.management_endpoints.team_endpoints import list_team

                teams_2 = await list_team(
                    http_request=Request(
                        scope={"type": "http", "path": "/user/info"},
                    ),
                    user_api_key_dict=user_api_key_dict,
                )
            elif caller_user_info is not None:
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
                    key = key.model_dump()  # noqa
                except Exception:
                    # if using pydantic v1
                    key = key.dict()
                if (
                    "team_id" in key
                    and key["team_id"] is not None
                    and key["team_id"] != "litellm-dashboard"
                ):
                    team_info = get_team_from_list(
                        team_list=teams_1, team_id=key["team_id"]
                    )
                    if team_info is not None:
                        team_alias = getattr(team_info, "team_alias", None)
                        key["team_alias"] = team_alias
                    else:
                        key["team_alias"] = None
                else:
                    key["team_alias"] = "None"
                returned_keys.append(key)

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

    See below for all params 
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        data_json: dict = data.json()
        # get the row from db
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        # get non default values for key
        non_default_values = {}
        for k, v in data_json.items():
            if v is not None and v not in (
                [],
                {},
                0,
            ):  # models default to [], spend defaults to 0, we should not reset these values
                non_default_values[k] = v

        is_internal_user = False
        if data.user_role == LitellmUserRoles.INTERNAL_USER:
            is_internal_user = True

        if "budget_duration" in non_default_values:
            duration_s = _duration_in_seconds(
                duration=non_default_values["budget_duration"]
            )
            user_reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)
            non_default_values["budget_reset_at"] = user_reset_at

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
                duration_s = _duration_in_seconds(
                    duration=non_default_values["budget_duration"]
                )
                user_reset_at = datetime.now(timezone.utc) + timedelta(
                    seconds=duration_s
                )
                non_default_values["budget_reset_at"] = user_reset_at

        ## ADD USER, IF NEW ##
        verbose_proxy_logger.debug("/user/update: Received data = %s", data)
        response: Optional[Any] = None
        if data.user_id is not None and len(data.user_id) > 0:
            non_default_values["user_id"] = data.user_id  # type: ignore
            verbose_proxy_logger.debug("In update user, user_id condition block.")
            response = await prisma_client.update_data(
                user_id=data.user_id,
                data=non_default_values,
                table_name="user",
            )
            verbose_proxy_logger.debug(
                f"received response from updating prisma client. response={response}"
            )
        elif data.user_email is not None:
            non_default_values["user_id"] = str(uuid.uuid4())
            non_default_values["user_email"] = data.user_email
            ## user email is not unique acc. to prisma schema -> future improvement
            ### for now: check if it exists in db, if not - insert it
            existing_user_rows = await prisma_client.get_data(
                key_val={"user_email": data.user_email},
                table_name="user",
                query_type="find_all",
            )
            if existing_user_rows is None or (
                isinstance(existing_user_rows, list) and len(existing_user_rows) == 0
            ):
                response = await prisma_client.insert_data(
                    data=non_default_values, table_name="user"
                )
            elif isinstance(existing_user_rows, list) and len(existing_user_rows) > 0:
                for existing_user in existing_user_rows:
                    response = await prisma_client.update_data(
                        user_id=existing_user.user_id,
                        data=non_default_values,
                        table_name="user",
                    )
        return response  # type: ignore
        # update based on remaining passed in values
    except Exception as e:
        verbose_proxy_logger.error(
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


@router.post(
    "/user/request_model",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def user_request_model(request: Request):
    """
    Allow a user to create a request to access a model
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        data_json = await request.json()

        # get the row from db
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        non_default_values = {k: v for k, v in data_json.items() if v is not None}
        new_models = non_default_values.get("models", None)
        user_id = non_default_values.get("user_id", None)
        justification = non_default_values.get("justification", None)

        await prisma_client.insert_data(
            data={
                "models": new_models,
                "justification": justification,
                "user_id": user_id,
                "status": "pending",
                "request_id": str(uuid.uuid4()),
            },
            table_name="user_notification",
        )
        return {"status": "success"}
        # update based on remaining passed in values
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.user_request_model(): Exception occured - {}".format(
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


@router.get(
    "/user/get_requests",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def user_get_requests():
    """
    Get all "Access" requests made by proxy users, access requests are requests for accessing models
    """
    from litellm.proxy.proxy_server import prisma_client

    try:

        # get the row from db
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        # TODO: Optimize this so we don't read all the data here, eventually move to pagination
        response = await prisma_client.get_data(
            query_type="find_all",
            table_name="user_notification",
        )
        return {"requests": response}
        # update based on remaining passed in values
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.user_get_requests(): Exception occured - {}".format(
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


@router.get(
    "/user/get_users",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.get(
    "/user/list",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_users(
    role: Optional[str] = fastapi.Query(
        default=None, description="Filter users by role"
    ),
    page: int = fastapi.Query(default=1, ge=1, description="Page number"),
    page_size: int = fastapi.Query(
        default=25, ge=1, le=100, description="Number of items per page"
    ),
):
    """
    Get a paginated list of users, optionally filtered by role.

    Used by the UI to populate the user lists.

    Currently - admin-only endpoint.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": f"No db connected. prisma client={prisma_client}"},
        )

    # Calculate skip and take for pagination
    skip = (page - 1) * page_size
    take = page_size

    # Prepare the query
    query = {}
    if role:
        query["user_role"] = role

    # Get total count
    total_count = await prisma_client.db.litellm_usertable.count(where=query)  # type: ignore

    # Get paginated users
    users = await prisma_client.db.litellm_usertable.find_many(
        where=query,  # type: ignore
        skip=skip,
        take=take,
    )

    # Calculate total pages
    total_pages = -(-total_count // page_size)  # Ceiling division

    return {
        "users": users,
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
    curl --location 'http://0.0.0.0:8000/user/delete' \

    --header 'Authorization: Bearer sk-1234' \

    --header 'Content-Type: application/json' \

    --data-raw '{
        "user_ids": ["45e3e396-ee08-4a61-a88e-16b3ce7e0849"]
    }'
    ```

    Parameters:
    - user_ids: List[str] - The list of user id's to be deleted.
    """
    from litellm.proxy.proxy_server import (
        _duration_in_seconds,
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
        user_api_key_cache,
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

    # End of Audit logging

    ## DELETE ASSOCIATED KEYS
    await prisma_client.db.litellm_verificationtoken.delete_many(
        where={"user_id": {"in": data.user_ids}}
    )

    ## DELETE ASSOCIATED INVITATION LINKS
    await prisma_client.db.litellm_invitationlink.delete_many(
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
