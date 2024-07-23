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

        team_callback_settings_obj.success_callback.append(data.callback_name)
    elif data.callback_type == "failure":
        if team_callback_settings_obj.failure_callback is None:
            team_callback_settings_obj.failure_callback = []
        team_callback_settings_obj.failure_callback.append(data.callback_name)
    elif data.callback_type == "success_and_failure":
        if team_callback_settings_obj.success_callback is None:
            team_callback_settings_obj.success_callback = []
        if team_callback_settings_obj.failure_callback is None:
            team_callback_settings_obj.failure_callback = []
        team_callback_settings_obj.success_callback.append(data.callback_name)
        team_callback_settings_obj.failure_callback.append(data.callback_name)
    for var, value in data.callback_vars.items():
        if team_callback_settings_obj.callback_vars is None:
            team_callback_settings_obj.callback_vars = {}
        team_callback_settings_obj.callback_vars[var] = value

    team_callback_settings_obj_dict = team_callback_settings_obj.model_dump()

    team_metadata["callback_settings"] = team_callback_settings_obj_dict
    team_metadata_json = json.dumps(team_metadata)  # update team_metadata

    await prisma_client.db.litellm_teamtable.update(
        where={"team_id": team_id}, data={"metadata": team_metadata_json}  # type: ignore
    )
