"""
Custom team endpoints that bypass premium checks for empty metadata fields.

This module provides a custom /team/update endpoint that doesn't fail when
premium fields (guardrails, logging, policies) are sent by non-premium users.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.proxy.management_endpoints.team_endpoints import UpdateTeamRequest

router = APIRouter()


@router.post(
    "/team/update", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def update_team_custom(  # noqa: PLR0915
    data: UpdateTeamRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Custom team update endpoint that bypasses premium checks for premium metadata fields.

    This endpoint accepts guardrails, logging, policies, and tags without requiring
    premium user status - allowing the frontend to update these fields freely.
    """
    # Lazy imports to avoid circular imports
    from litellm.proxy.management_endpoints.team_endpoints import (
        LiteLLM_TeamTable,
        LiteLLM_TeamTableCachedObj,
        handle_exception_on_proxy,
        _set_budget_reset_at,
        TeamMemberBudgetHandler,
        handle_update_object_permission,
        safe_dumps,
    )
    from litellm.proxy.auth.auth_checks import _cache_team_object
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    try:
        verbose_proxy_logger.info(
            "CUSTOM TEAM UPDATE ENDPOINT CALLED - team_id: %s, guardrails: %s, policies: %s, tags: %s",
            data.team_id,
            getattr(data, 'guardrails', None),
            getattr(data, 'policies', None),
            getattr(data, 'tags', None),
        )

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": "Not connected to DB"},
            )

        if data.team_id is None:
            raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

        verbose_proxy_logger.debug("/team/update (custom) - %s", data)

        # Validate budget values are not negative
        if data.max_budget is not None and data.max_budget < 0:
            raise HTTPException(
                status_code=400,
                detail={"error": f"max_budget cannot be negative. Received: {data.max_budget}"}
            )
        if data.team_member_budget is not None and data.team_member_budget < 0:
            raise HTTPException(
                status_code=400,
                detail={"error": f"team_member_budget cannot be negative. Received: {data.team_member_budget}"}
            )

        existing_team_row = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": data.team_id}
        )

        if existing_team_row is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team not found, passed team_id={data.team_id}"},
            )

        # Get the data as a dict, excluding unset fields
        updated_kv = data.json(exclude_unset=True)

        # Check budget_duration and budget_reset_at
        _set_budget_reset_at(data, updated_kv)

        if TeamMemberBudgetHandler.should_create_budget(
            team_member_budget=data.team_member_budget,
            team_member_rpm_limit=data.team_member_rpm_limit,
            team_member_tpm_limit=data.team_member_tpm_limit,
            team_member_budget_duration=data.team_member_budget_duration,
        ):
            updated_kv = await TeamMemberBudgetHandler.upsert_team_member_budget_table(
                team_table=existing_team_row,
                user_api_key_dict=user_api_key_dict,
                updated_kv=updated_kv,
                team_member_budget=data.team_member_budget,
                team_member_rpm_limit=data.team_member_rpm_limit,
                team_member_tpm_limit=data.team_member_tpm_limit,
                team_member_budget_duration=data.team_member_budget_duration,
            )
        else:
            TeamMemberBudgetHandler._clean_team_member_fields(updated_kv)

        # Check object permission
        if data.object_permission is not None:
            updated_kv = await handle_update_object_permission(
                data_json=updated_kv,
                existing_team_row=existing_team_row,
            )

        # SKIPPED: _update_metadata_fields - This has premium checks that we want to bypass
        # The premium fields (guardrails, policies, tags, etc.) are saved directly to the team table
        # without requiring premium user status

        # Handle model_aliases
        if "model_aliases" in updated_kv:
            updated_kv.pop("model_aliases")

        # Serialize router_settings to JSON if present
        if "router_settings" in updated_kv and updated_kv["router_settings"] is not None:
            updated_kv["router_settings"] = safe_dumps(updated_kv["router_settings"])

        updated_kv = prisma_client.jsonify_team_object(db_data=updated_kv)

        team_row: Optional[LiteLLM_TeamTable] = (
            await prisma_client.db.litellm_teamtable.update(
                where={"team_id": data.team_id},
                data=updated_kv,
                include={"litellm_model_table": True},
            )
        )

        if team_row is None or team_row.team_id is None:
            raise HTTPException(
                status_code=400,
                detail={"error": "Team doesn't exist. Got={}".format(team_row)},
            )

        verbose_proxy_logger.info(
            "CUSTOM TEAM UPDATE SUCCESS - team_id: %s, guardrails: %s, policies: %s, tags: %s",
            team_row.team_id,
            getattr(team_row, 'guardrails', None),
            getattr(team_row, 'policies', None),
            getattr(team_row, 'tags', None),
        )

        await _cache_team_object(
            team_id=team_row.team_id,
            team_table=LiteLLM_TeamTableCachedObj(**team_row.model_dump()),
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        return {"team_id": team_row.team_id, "data": team_row}

    except HTTPException:
        raise
    except Exception as e:
        raise handle_exception_on_proxy(e)
