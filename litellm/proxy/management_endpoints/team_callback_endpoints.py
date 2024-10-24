"""
Endpoints to control callbacks per team

Use this when each team should control its own callbacks
"""

import asyncio
import copy
import json
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    AddTeamCallback,
    LiteLLM_TeamTable,
    ProxyErrorTypes,
    ProxyException,
    TeamCallbackMetadata,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.utils import (
    add_new_member,
    management_endpoint_wrapper,
)

router = APIRouter()


@router.post(
    "/team/{team_id:path}/callback",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def add_team_callbacks(
    data: AddTeamCallback,
    http_request: Request,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Add a success/failure callback to a team

    Use this if if you want different teams to have different success/failure callbacks

    Example curl:
    ```
    curl -X POST 'http:/localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback' \
        -H 'Content-Type: application/json' \
        -H 'Authorization: Bearer sk-1234' \
        -d '{
        "callback_name": "langfuse",
        "callback_type": "success",
        "callback_vars": {"langfuse_public_key": "pk-lf-xxxx1", "langfuse_secret_key": "sk-xxxxx"}
        
    }'
    ```

    This means for the team where team_id = dbe2f686-a686-4896-864a-4c3924458709, all LLM calls will be logged to langfuse using the public key pk-lf-xxxx1 and the secret key sk-xxxxx

    """
    try:
        from litellm.proxy.proxy_server import (
            _duration_in_seconds,
            create_audit_log_for_update,
            litellm_proxy_admin_name,
            prisma_client,
        )

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

        # Check if team_id exists already
        _existing_team = await prisma_client.get_data(
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if _existing_team is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Team id = {team_id} does not exist. Please use a different team id."
                },
            )

        # store team callback settings in metadata
        team_metadata = _existing_team.metadata
        team_callback_settings = team_metadata.get("callback_settings", {})
        # expect callback settings to be
        team_callback_settings_obj = TeamCallbackMetadata(**team_callback_settings)
        if data.callback_type == "success":
            if team_callback_settings_obj.success_callback is None:
                team_callback_settings_obj.success_callback = []

            if data.callback_name in team_callback_settings_obj.success_callback:
                raise ProxyException(
                    message=f"callback_name = {data.callback_name} already exists in failure_callback, for team_id = {team_id}. \n Existing failure_callback = {team_callback_settings_obj.success_callback}",
                    code=status.HTTP_400_BAD_REQUEST,
                    type=ProxyErrorTypes.bad_request_error,
                    param="callback_name",
                )

            team_callback_settings_obj.success_callback.append(data.callback_name)
        elif data.callback_type == "failure":
            if team_callback_settings_obj.failure_callback is None:
                team_callback_settings_obj.failure_callback = []

            if data.callback_name in team_callback_settings_obj.failure_callback:
                raise ProxyException(
                    message=f"callback_name = {data.callback_name} already exists in failure_callback, for team_id = {team_id}. \n Existing failure_callback = {team_callback_settings_obj.failure_callback}",
                    code=status.HTTP_400_BAD_REQUEST,
                    type=ProxyErrorTypes.bad_request_error,
                    param="callback_name",
                )
            team_callback_settings_obj.failure_callback.append(data.callback_name)
        elif data.callback_type == "success_and_failure":
            if team_callback_settings_obj.success_callback is None:
                team_callback_settings_obj.success_callback = []
            if team_callback_settings_obj.failure_callback is None:
                team_callback_settings_obj.failure_callback = []
            if data.callback_name in team_callback_settings_obj.success_callback:
                raise ProxyException(
                    message=f"callback_name = {data.callback_name} already exists in success_callback, for team_id = {team_id}. \n Existing success_callback = {team_callback_settings_obj.success_callback}",
                    code=status.HTTP_400_BAD_REQUEST,
                    type=ProxyErrorTypes.bad_request_error,
                    param="callback_name",
                )

            if data.callback_name in team_callback_settings_obj.failure_callback:
                raise ProxyException(
                    message=f"callback_name = {data.callback_name} already exists in failure_callback, for team_id = {team_id}. \n Existing failure_callback = {team_callback_settings_obj.failure_callback}",
                    code=status.HTTP_400_BAD_REQUEST,
                    type=ProxyErrorTypes.bad_request_error,
                    param="callback_name",
                )

            team_callback_settings_obj.success_callback.append(data.callback_name)
            team_callback_settings_obj.failure_callback.append(data.callback_name)
        for var, value in data.callback_vars.items():
            if team_callback_settings_obj.callback_vars is None:
                team_callback_settings_obj.callback_vars = {}
            team_callback_settings_obj.callback_vars[var] = value

        team_callback_settings_obj_dict = team_callback_settings_obj.model_dump()

        team_metadata["callback_settings"] = team_callback_settings_obj_dict
        team_metadata_json = json.dumps(team_metadata)  # update team_metadata

        new_team_row = await prisma_client.db.litellm_teamtable.update(
            where={"team_id": team_id}, data={"metadata": team_metadata_json}  # type: ignore
        )

        return {
            "status": "success",
            "data": new_team_row,
        }

    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.add_team_callbacks(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
                type=ProxyErrorTypes.internal_server_error.value,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error.value,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post(
    "/team/{team_id}/disable_logging",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def disable_team_logging(
    http_request: Request,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

        # Check if team exists
        _existing_team = await prisma_client.get_data(
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if _existing_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist."},
            )

        # Update team metadata to disable logging
        team_metadata = _existing_team.metadata
        team_callback_settings = team_metadata.get("callback_settings", {})
        team_callback_settings_obj = TeamCallbackMetadata(**team_callback_settings)

        # Reset callbacks
        team_callback_settings_obj.success_callback = []
        team_callback_settings_obj.failure_callback = []

        # Update metadata
        team_metadata["callback_settings"] = team_callback_settings_obj.model_dump()
        team_metadata_json = json.dumps(team_metadata)

        # Update team in database
        updated_team = await prisma_client.db.litellm_teamtable.update(
            where={"team_id": team_id}, data={"metadata": team_metadata_json}  # type: ignore
        )

        if updated_team is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Team id = {team_id} does not exist. Error updating team logging"
                },
            )

        return {
            "status": "success",
            "message": f"Logging disabled for team {team_id}",
            "data": {
                "team_id": updated_team.team_id,
                "success_callbacks": [],
                "failure_callbacks": [],
            },
        }

    except Exception as e:
        verbose_proxy_logger.error(
            f"litellm.proxy.proxy_server.disable_team_logging(): Exception occurred - {str(e)}"
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
                type=ProxyErrorTypes.internal_server_error.value,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error.value,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/team/{team_id:path}/callback",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def get_team_callbacks(
    http_request: Request,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get the success/failure callbacks and variables for a team

    Example curl:
    ```
    curl -X GET 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback' \
        -H 'Authorization: Bearer sk-1234'
    ```

    This will return the callback settings for the team with id dbe2f686-a686-4896-864a-4c3924458709

    Returns {
            "status": "success",
            "data": {
                "team_id": team_id,
                "success_callbacks": team_callback_settings_obj.success_callback,
                "failure_callbacks": team_callback_settings_obj.failure_callback,
                "callback_vars": team_callback_settings_obj.callback_vars,
            },
        }
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

        # Check if team_id exists
        _existing_team = await prisma_client.get_data(
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if _existing_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist."},
            )

        # Retrieve team callback settings from metadata
        team_metadata = _existing_team.metadata
        team_callback_settings = team_metadata.get("callback_settings", {})

        # Convert to TeamCallbackMetadata object for consistent structure
        team_callback_settings_obj = TeamCallbackMetadata(**team_callback_settings)

        return {
            "status": "success",
            "data": {
                "team_id": team_id,
                "success_callbacks": team_callback_settings_obj.success_callback,
                "failure_callbacks": team_callback_settings_obj.failure_callback,
                "callback_vars": team_callback_settings_obj.callback_vars,
            },
        }

    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.get_team_callbacks(): Exception occurred - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
                type=ProxyErrorTypes.internal_server_error.value,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error.value,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
