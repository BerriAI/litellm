"""
Custom team endpoints that bypass premium checks for team metadata fields.

The standard v1.92 /team/update path is kept as the source of truth for
validation, object-permission handling, cache refresh, and audit logging. This
router only pre-moves premium metadata fields into the metadata JSON blob before
delegating, which avoids the premium check while preserving the maintained code
path.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, Request

from litellm.proxy._types import (
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.team_endpoints import (
    UpdateTeamRequest,
    update_team,
)
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper


router = APIRouter()


def _team_update_without_premium_top_level_fields(
    data: UpdateTeamRequest,
) -> UpdateTeamRequest:
    data_json: Dict[str, Any] = data.json(exclude_unset=True)
    metadata = data_json.get("metadata")
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        metadata = dict(metadata)

    moved_premium_metadata = False
    for field in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
        if field in data_json and data_json[field] is not None:
            metadata[field] = data_json.pop(field)
            moved_premium_metadata = True

    if moved_premium_metadata:
        data_json["metadata"] = metadata

    return UpdateTeamRequest(**data_json)


@router.post("/team/update", tags=["team management"], dependencies=[Depends(user_api_key_auth)])
@management_endpoint_wrapper
async def update_team_custom(
    data: UpdateTeamRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    delegated_update_team = getattr(update_team, "__wrapped__", update_team)
    return await delegated_update_team(
        data=_team_update_without_premium_top_level_fields(data),
        http_request=http_request,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=litellm_changed_by,
    )
