"""
TEAM MANAGEMENT

All /team management endpoints

/team/new
/team/info
/team/update
/team/delete
"""

import asyncio
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import (
    BlockTeamRequest,
    CommonProxyErrors,
    DeleteTeamRequest,
    LiteLLM_AuditLogs,
    LiteLLM_ManagementEndpoint_MetadataFields,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    LiteLLM_ModelTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmTableNames,
    LitellmUserRoles,
    Member,
    NewTeamRequest,
    ProxyErrorTypes,
    ProxyException,
    SpecialManagementEndpointEnums,
    SpecialModelNames,
    SpecialProxyStrings,
    TeamAddMemberResponse,
    TeamInfoResponseObject,
    TeamInfoResponseObjectTeamTable,
    TeamListResponseObject,
    TeamMemberAddRequest,
    TeamMemberDeleteRequest,
    TeamMemberUpdateRequest,
    TeamMemberUpdateResponse,
    TeamModelAddRequest,
    TeamModelDeleteRequest,
    UpdateTeamRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    allowed_route_check_inside_route,
    can_org_access_model,
    get_org_object,
    get_team_object,
    get_user_object,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_team_admin,
    _set_object_metadata_field,
    _update_metadata_field,
    _upsert_budget_and_membership,
    _user_has_admin_view,
)
from litellm.proxy.management_endpoints.tag_management_endpoints import (
    get_daily_activity,
)
from litellm.proxy.management_helpers.object_permission_utils import (
    _set_object_permission,
    handle_update_object_permission_common,
)
from litellm.proxy.management_helpers.team_member_permission_checks import (
    TeamMemberPermissionChecks,
)
from litellm.proxy.management_helpers.utils import (
    add_new_member,
    management_endpoint_wrapper,
)
from litellm.proxy.utils import PrismaClient, handle_exception_on_proxy
from litellm.router import Router
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.management_endpoints.team_endpoints import (
    BulkTeamMemberAddRequest,
    BulkTeamMemberAddResponse,
    GetTeamMemberPermissionsResponse,
    TeamListResponse,
    TeamMemberAddResult,
    UpdateTeamMemberPermissionsRequest,
)

router = APIRouter()


class TeamMemberBudgetHandler:
    """Helper class to handle team member budget, RPM, and TPM limit operations"""

    @staticmethod
    def should_create_budget(
        team_member_budget: Optional[float] = None,
        team_member_rpm_limit: Optional[int] = None,
        team_member_tpm_limit: Optional[int] = None,
    ) -> bool:
        """Check if any team member limits are provided"""
        return any(
            [
                team_member_budget is not None,
                team_member_rpm_limit is not None,
                team_member_tpm_limit is not None,
            ]
        )

    @staticmethod
    async def create_team_member_budget_table(
        data: Union[NewTeamRequest, LiteLLM_TeamTable],
        new_team_data_json: dict,
        user_api_key_dict: UserAPIKeyAuth,
        team_member_budget: Optional[float] = None,
        team_member_rpm_limit: Optional[int] = None,
        team_member_tpm_limit: Optional[int] = None,
    ) -> dict:
        """Create team member budget table with provided limits"""
        from litellm.proxy._types import BudgetNewRequest
        from litellm.proxy.management_endpoints.budget_management_endpoints import (
            new_budget,
        )

        if data.team_alias is not None:
            budget_id = (
                f"team-{data.team_alias.replace(' ', '-')}-budget-{uuid.uuid4().hex}"
            )
        else:
            budget_id = f"team-budget-{uuid.uuid4().hex}"

        # Create budget request with all provided limits
        budget_request = BudgetNewRequest(
            budget_id=budget_id,
            budget_duration=data.budget_duration,
        )

        if team_member_budget is not None:
            budget_request.max_budget = team_member_budget
        if team_member_rpm_limit is not None:
            budget_request.rpm_limit = team_member_rpm_limit
        if team_member_tpm_limit is not None:
            budget_request.tpm_limit = team_member_tpm_limit

        team_member_budget_table = await new_budget(
            budget_obj=budget_request,
            user_api_key_dict=user_api_key_dict,
        )

        # Add team_member_budget_id as metadata field to team table
        if new_team_data_json.get("metadata") is None:
            new_team_data_json["metadata"] = {}
        new_team_data_json["metadata"][
            "team_member_budget_id"
        ] = team_member_budget_table.budget_id

        # Remove team member fields from new_team_data_json
        TeamMemberBudgetHandler._clean_team_member_fields(new_team_data_json)

        return new_team_data_json

    @staticmethod
    async def upsert_team_member_budget_table(
        team_table: LiteLLM_TeamTable,
        user_api_key_dict: UserAPIKeyAuth,
        updated_kv: dict,
        team_member_budget: Optional[float] = None,
        team_member_rpm_limit: Optional[int] = None,
        team_member_tpm_limit: Optional[int] = None,
    ) -> dict:
        """Upsert team member budget table with provided limits"""
        from litellm.proxy._types import BudgetNewRequest
        from litellm.proxy.management_endpoints.budget_management_endpoints import (
            update_budget,
        )

        if team_table.metadata is None:
            team_table.metadata = {}

        team_member_budget_id = team_table.metadata.get("team_member_budget_id")
        if team_member_budget_id is not None and isinstance(team_member_budget_id, str):
            # Budget exists - create update request with only provided values
            budget_request = BudgetNewRequest(budget_id=team_member_budget_id)

            if team_member_budget is not None:
                budget_request.max_budget = team_member_budget
            if team_member_rpm_limit is not None:
                budget_request.rpm_limit = team_member_rpm_limit
            if team_member_tpm_limit is not None:
                budget_request.tpm_limit = team_member_tpm_limit

            budget_row = await update_budget(
                budget_obj=budget_request,
                user_api_key_dict=user_api_key_dict,
            )
            verbose_proxy_logger.info(
                f"Updated team member budget table: {budget_row.budget_id}, with team_member_budget={team_member_budget}, team_member_rpm_limit={team_member_rpm_limit}, team_member_tpm_limit={team_member_tpm_limit}"
            )
            if updated_kv.get("metadata") is None:
                updated_kv["metadata"] = {}
            updated_kv["metadata"]["team_member_budget_id"] = budget_row.budget_id

        else:  # budget does not exist
            updated_kv = await TeamMemberBudgetHandler.create_team_member_budget_table(
                data=team_table,
                new_team_data_json=updated_kv,
                user_api_key_dict=user_api_key_dict,
                team_member_budget=team_member_budget,
                team_member_rpm_limit=team_member_rpm_limit,
                team_member_tpm_limit=team_member_tpm_limit,
            )

        # Remove team member fields from updated_kv
        TeamMemberBudgetHandler._clean_team_member_fields(updated_kv)
        return updated_kv

    @staticmethod
    def _clean_team_member_fields(data_dict: dict) -> None:
        """Remove team member fields from data dictionary"""
        data_dict.pop("team_member_budget", None)
        data_dict.pop("team_member_rpm_limit", None)
        data_dict.pop("team_member_tpm_limit", None)


def _is_available_team(team_id: str, user_api_key_dict: UserAPIKeyAuth) -> bool:
    if litellm.default_internal_user_params is None:
        return False
    if "available_teams" in litellm.default_internal_user_params:
        return team_id in litellm.default_internal_user_params["available_teams"]
    return False


async def get_all_team_memberships(
    prisma_client: PrismaClient, team_ids: List[str], user_id: Optional[str] = None
) -> List[LiteLLM_TeamMembership]:
    """Get all team memberships for a given user"""
    ## GET ALL MEMBERSHIPS ##
    where_obj: Dict[str, Dict[str, List[str]]] = {"team_id": {"in": team_ids}}
    if user_id is not None:
        where_obj["user_id"] = {"in": [user_id]}
    # if user_id is None:
    #     where_obj = {"team_id": {"in": team_id}}
    # else:
    #     where_obj = {"user_id": str(user_id), "team_id": {"in": team_id}}

    team_memberships = await prisma_client.db.litellm_teammembership.find_many(
        where=where_obj,
        include={"litellm_budget_table": True},
    )

    returned_tm: List[LiteLLM_TeamMembership] = []
    for tm in team_memberships:
        returned_tm.append(LiteLLM_TeamMembership(**tm.model_dump()))

    return returned_tm


def _check_team_model_specific_limits(
    teams: List[LiteLLM_TeamTable],
    data: Union[NewTeamRequest, UpdateTeamRequest],
    entity_rpm_limit: Optional[int],
    entity_tpm_limit: Optional[int],
    entity_model_rpm_limit_dict: Dict[str, int],
    entity_model_tpm_limit_dict: Dict[str, int],
    entity_type: str,  # "organization"
) -> None:
    """
    Generic function to check if a team is allocating model specific limits.
    Raises an error if we're overallocating.
    """
    model_rpm_limit = getattr(data, "model_rpm_limit", None) or (
        data.metadata.get("model_rpm_limit", None) if data.metadata else None
    )
    model_tpm_limit = getattr(data, "model_tpm_limit", None) or (
        data.metadata.get("model_tpm_limit", None) if data.metadata else None
    )
    if model_rpm_limit is None and model_tpm_limit is None:
        return

    # get total model specific tpm/rpm limit
    model_specific_rpm_limit: Dict[str, int] = {}
    model_specific_tpm_limit: Dict[str, int] = {}

    for team in teams:
        if team.metadata and team.metadata.get("model_rpm_limit", None) is not None:
            for model, rpm_limit in team.metadata.get("model_rpm_limit", {}).items():
                model_specific_rpm_limit[model] = (
                    model_specific_rpm_limit.get(model, 0) + rpm_limit
                )
        if team.metadata and team.metadata.get("model_tpm_limit", None) is not None:
            for model, tpm_limit in team.metadata.get("model_tpm_limit", {}).items():
                model_specific_tpm_limit[model] = (
                    model_specific_tpm_limit.get(model, 0) + tpm_limit
                )

    if model_rpm_limit is not None:
        for model, rpm_limit in model_rpm_limit.items():
            if (
                entity_rpm_limit is not None
                and model_specific_rpm_limit.get(model, 0) + rpm_limit
                > entity_rpm_limit
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Allocated RPM limit={model_specific_rpm_limit.get(model, 0)} + Team RPM limit={rpm_limit} is greater than {entity_type} RPM limit={entity_rpm_limit}",
                )
            elif entity_model_rpm_limit_dict:
                entity_model_specific_rpm_limit = entity_model_rpm_limit_dict.get(model)
                if (
                    entity_model_specific_rpm_limit
                    and model_specific_rpm_limit.get(model, 0) + rpm_limit
                    > entity_model_specific_rpm_limit
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Allocated RPM limit={model_specific_rpm_limit.get(model, 0)} + Team RPM limit={rpm_limit} is greater than {entity_type} RPM limit={entity_model_specific_rpm_limit}",
                    )

    if model_tpm_limit is not None:
        for model, tpm_limit in model_tpm_limit.items():
            if (
                entity_tpm_limit is not None
                and model_specific_tpm_limit.get(model, 0) + tpm_limit
                > entity_tpm_limit
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Allocated TPM limit={model_specific_tpm_limit.get(model, 0)} + Team TPM limit={tpm_limit} is greater than {entity_type} TPM limit={entity_tpm_limit}",
                )
            elif entity_model_tpm_limit_dict:
                entity_model_specific_tpm_limit = entity_model_tpm_limit_dict.get(model)
                if (
                    entity_model_specific_tpm_limit
                    and model_specific_tpm_limit.get(model, 0) + tpm_limit
                    > entity_model_specific_tpm_limit
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Allocated TPM limit={model_specific_tpm_limit.get(model, 0)} + Team TPM limit={tpm_limit} is greater than {entity_type} TPM limit={entity_model_specific_tpm_limit}",
                    )


def _check_team_rpm_tpm_limits(
    teams: List[LiteLLM_TeamTable],
    data: Union[NewTeamRequest, UpdateTeamRequest],
    entity_rpm_limit: Optional[int],
    entity_tpm_limit: Optional[int],
    entity_type: str,  # "organization"
) -> None:
    """
    Generic function to check if a team is allocating rpm/tpm limits.
    Raises an error if we're overallocating.
    """
    if teams is not None and len(teams) > 0:
        allocated_tpm = sum(
            team.tpm_limit for team in teams if team.tpm_limit is not None
        )
        allocated_rpm = sum(
            team.rpm_limit for team in teams if team.rpm_limit is not None
        )
    else:
        allocated_tpm = 0
        allocated_rpm = 0

    if (
        data.tpm_limit is not None
        and entity_tpm_limit is not None
        and data.tpm_limit + allocated_tpm > entity_tpm_limit
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Allocated TPM limit={allocated_tpm} + Team TPM limit={data.tpm_limit} is greater than {entity_type} TPM limit={entity_tpm_limit}",
        )
    if (
        data.rpm_limit is not None
        and entity_rpm_limit is not None
        and data.rpm_limit + allocated_rpm > entity_rpm_limit
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Allocated RPM limit={allocated_rpm} + Team RPM limit={data.rpm_limit} is greater than {entity_type} RPM limit={entity_rpm_limit}",
        )


def check_org_team_model_specific_limits(
    teams: List[LiteLLM_TeamTable],
    org_table: LiteLLM_OrganizationTable,
    data: Union[NewTeamRequest, UpdateTeamRequest],
) -> None:
    """
    Check if the organization team is allocating model specific limits. If so, raise an error if we're overallocating.
    """

    # Get org limits from budget table if available
    entity_rpm_limit = None
    entity_tpm_limit = None
    entity_model_rpm_limit_dict = {}
    entity_model_tpm_limit_dict = {}

    if org_table.litellm_budget_table is not None:
        entity_rpm_limit = org_table.litellm_budget_table.rpm_limit
        entity_tpm_limit = org_table.litellm_budget_table.tpm_limit

    if org_table.metadata:
        entity_model_rpm_limit_dict = org_table.metadata.get("model_rpm_limit", {})
        entity_model_tpm_limit_dict = org_table.metadata.get("model_tpm_limit", {})

    _check_team_model_specific_limits(
        teams=teams,
        data=data,
        entity_rpm_limit=entity_rpm_limit,
        entity_tpm_limit=entity_tpm_limit,
        entity_model_rpm_limit_dict=entity_model_rpm_limit_dict,
        entity_model_tpm_limit_dict=entity_model_tpm_limit_dict,
        entity_type="organization",
    )


def check_org_team_rpm_tpm_limits(
    teams: List[LiteLLM_TeamTable],
    org_table: LiteLLM_OrganizationTable,
    data: Union[NewTeamRequest, UpdateTeamRequest],
) -> None:
    """
    Check if the organization team is allocating rpm/tpm limits. If so, raise an error if we're overallocating.
    """
    # Get org limits from budget table if available
    entity_rpm_limit = None
    entity_tpm_limit = None

    if org_table.litellm_budget_table is not None:
        entity_rpm_limit = org_table.litellm_budget_table.rpm_limit
        entity_tpm_limit = org_table.litellm_budget_table.tpm_limit

    _check_team_rpm_tpm_limits(
        teams=teams,
        data=data,
        entity_rpm_limit=entity_rpm_limit,
        entity_tpm_limit=entity_tpm_limit,
        entity_type="organization",
    )


async def _check_org_team_limits(
    org_table: LiteLLM_OrganizationTable,
    data: Union[NewTeamRequest, UpdateTeamRequest],
    prisma_client: PrismaClient,
) -> None:
    """
    Check if the organization team is allocating guaranteed throughput limits. If so, raise an error if we're overallocating.

    Only runs check if tpm_limit_type or rpm_limit_type is "guaranteed_throughput"
    """

    rpm_limit_type = getattr(data, "rpm_limit_type", None) or (
        data.metadata.get("rpm_limit_type", None) if data.metadata else None
    )
    tpm_limit_type = getattr(data, "tpm_limit_type", None) or (
        data.metadata.get("tpm_limit_type", None) if data.metadata else None
    )

    if (
        tpm_limit_type != "guaranteed_throughput"
        and rpm_limit_type != "guaranteed_throughput"
    ):
        return
    # get all organization teams
    # calculate allocated tpm/rpm limit
    # check if specified tpm/rpm limit is greater than allocated tpm/rpm limit

    teams = await prisma_client.db.litellm_teamtable.find_many(
        where={"organization_id": org_table.organization_id},
    )

    # Convert teams to LiteLLM_TeamTable objects
    team_objs: List[LiteLLM_TeamTable] = []
    for team in teams:
        team_objs.append(LiteLLM_TeamTable(**team.model_dump()))

    check_org_team_model_specific_limits(
        teams=team_objs,
        org_table=org_table,
        data=data,
    )
    check_org_team_rpm_tpm_limits(
        teams=team_objs,
        org_table=org_table,
        data=data,
    )


#### TEAM MANAGEMENT ####
@router.post(
    "/team/new",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_TeamTable,
)
@management_endpoint_wrapper
async def new_team(  # noqa: PLR0915
    data: NewTeamRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Allow users to create a new team. Apply user permissions to their team.

    👉 [Detailed Doc on setting team budgets](https://docs.litellm.ai/docs/proxy/team_budgets)


    Parameters:
    - team_alias: Optional[str] - User defined team alias
    - team_id: Optional[str] - The team id of the user. If none passed, we'll generate it.
    - members_with_roles: List[{"role": "admin" or "user", "user_id": "<user-id>"}] - A list of users and their roles in the team. Get user_id when making a new user via `/user/new`.
    - team_member_permissions: Optional[List[str]] - A list of routes that non-admin team members can access. example: ["/key/generate", "/key/update", "/key/delete"]
    - metadata: Optional[dict] - Metadata for team, store information for team. Example metadata = {"extra_info": "some info"}
    - model_rpm_limit: Optional[Dict[str, int]] - The RPM (Requests Per Minute) limit for this team - applied across all keys for this team. 
    - model_tpm_limit: Optional[Dict[str, int]] - The TPM (Tokens Per Minute) limit for this team - applied across all keys for this team.
    - tpm_limit: Optional[int] - The TPM (Tokens Per Minute) limit for this team - all keys with this team_id will have at max this TPM limit
    - rpm_limit: Optional[int] - The RPM (Requests Per Minute) limit for this team - all keys associated with this team_id will have at max this RPM limit
    - rpm_limit_type: Optional[Literal["guaranteed_throughput", "best_effort_throughput"]] - The type of RPM limit enforcement. Use "guaranteed_throughput" to raise an error if overallocating RPM, or "best_effort_throughput" for best effort enforcement.
    - tpm_limit_type: Optional[Literal["guaranteed_throughput", "best_effort_throughput"]] - The type of TPM limit enforcement. Use "guaranteed_throughput" to raise an error if overallocating TPM, or "best_effort_throughput" for best effort enforcement.
    - max_budget: Optional[float] - The maximum budget allocated to the team - all keys for this team_id will have at max this max_budget
    - budget_duration: Optional[str] - The duration of the budget for the team. Doc [here](https://docs.litellm.ai/docs/proxy/team_budgets)
    - models: Optional[list] - A list of models associated with the team - all keys for this team_id will have at most, these models. If empty, assumes all models are allowed.
    - blocked: bool - Flag indicating if the team is blocked or not - will stop all calls from keys with this team_id.
    - members: Optional[List] - Control team members via `/team/member/add` and `/team/member/delete`.
    - tags: Optional[List[str]] - Tags for [tracking spend](https://litellm.vercel.app/docs/proxy/enterprise#tracking-spend-for-custom-tags) and/or doing [tag-based routing](https://litellm.vercel.app/docs/proxy/tag_routing).
    - prompts: Optional[List[str]] - List of prompts that the team is allowed to use.
    - organization_id: Optional[str] - The organization id of the team. Default is None. Create via `/organization/new`.
    - model_aliases: Optional[dict] - Model aliases for the team. [Docs](https://docs.litellm.ai/docs/proxy/team_based_routing#create-team-with-model-alias)
    - guardrails: Optional[List[str]] - Guardrails for the team. [Docs](https://docs.litellm.ai/docs/proxy/guardrails)
    - prompts: Optional[List[str]] - List of prompts that the team is allowed to use.
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - team-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"], "mcp_tool_permissions": {"server_id_1": ["tool1", "tool2"]}}. IF null or {} then no object permission.
    - team_member_budget: Optional[float] - The maximum budget allocated to an individual team member.
    - team_member_rpm_limit: Optional[int] - The RPM (Requests Per Minute) limit for individual team members.
    - team_member_tpm_limit: Optional[int] - The TPM (Tokens Per Minute) limit for individual team members.
    - team_member_key_duration: Optional[str] - The duration for a team member's key. e.g. "1d", "1w", "1mo"
    - prompts: Optional[List[str]] - List of allowed prompts for the team. If specified, the team will only be able to use these specific prompts.
    - allowed_passthrough_routes: Optional[List[str]] - List of allowed pass through routes for the team.
    - allowed_vector_store_indexes: Optional[List[dict]] - List of allowed vector store indexes for the key. Example - [{"index_name": "my-index", "index_permissions": ["write", "read"]}]. If specified, the key will only be able to use these specific vector store indexes. Create index, using `/v1/indexes` endpoint.

    

    Returns:
    - team_id: (str) Unique team id - used for tracking spend across multiple keys for same team id.

    _deprecated_params:
    - admins: list - A list of user_id's for the admin role
    - users: list - A list of user_id's for the user role

    Example Request:
    ```
    curl --location 'http://0.0.0.0:4000/team/new' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
      "team_alias": "my-new-team_2",
      "members_with_roles": [{"role": "admin", "user_id": "user-1234"},
        {"role": "user", "user_id": "user-2434"}]
    }'

    ```

     ```
    curl --location 'http://0.0.0.0:4000/team/new' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
                "team_alias": "QA Prod Bot",
                "max_budget": 0.000000001,
                "budget_duration": "1d"
            }'
    ```
    """
    try:
        from litellm.proxy.proxy_server import (
            _license_check,
            create_audit_log_for_update,
            litellm_proxy_admin_name,
            prisma_client,
            user_api_key_cache,
        )

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

        # Check if license is over limit
        total_teams = await prisma_client.db.litellm_teamtable.count()
        if total_teams and _license_check.is_team_count_over_limit(
            team_count=total_teams
        ):
            raise HTTPException(
                status_code=403,
                detail="License is over limit. Please contact support@berri.ai to upgrade your license.",
            )

        if data.team_id is None:
            data.team_id = str(uuid.uuid4())
        else:
            # Check if team_id exists already
            _existing_team_id = await prisma_client.get_data(
                team_id=data.team_id, table_name="team", query_type="find_unique"
            )
            if _existing_team_id is not None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"Team id = {data.team_id} already exists. Please use a different team id."
                    },
                )

        # check org key limits - done here to handle inheriting org id from team
        if data.organization_id is not None and prisma_client is not None:
            org_table = await get_org_object(
                org_id=data.organization_id,
                user_api_key_cache=user_api_key_cache,
                prisma_client=prisma_client,
            )
            if org_table is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Organization not found for organization_id={data.organization_id}",
                )

            await _check_org_team_limits(
                org_table=org_table,
                data=data,
                prisma_client=prisma_client,
            )

        # If max_budget is not explicitly provided in the request,
        # check for a default value in the proxy configuration.
        if data.max_budget is None:
            if (
                isinstance(litellm.default_team_settings, list)
                and len(litellm.default_team_settings) > 0
                and isinstance(litellm.default_team_settings[0], dict)
            ):
                default_settings = litellm.default_team_settings[0]
                default_budget = default_settings.get("max_budget")
                if default_budget is not None:
                    data.max_budget = default_budget

        if (
            user_api_key_dict.user_role is None
            or user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        ):  # don't restrict proxy admin
            if (
                data.tpm_limit is not None
                and user_api_key_dict.tpm_limit is not None
                and data.tpm_limit > user_api_key_dict.tpm_limit
            ):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"tpm limit higher than user max. User tpm limit={user_api_key_dict.tpm_limit}. User role={user_api_key_dict.user_role}"
                    },
                )

            if (
                data.rpm_limit is not None
                and user_api_key_dict.rpm_limit is not None
                and data.rpm_limit > user_api_key_dict.rpm_limit
            ):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"rpm limit higher than user max. User rpm limit={user_api_key_dict.rpm_limit}. User role={user_api_key_dict.user_role}"
                    },
                )


            if (data.max_budget is not None and user_api_key_dict.user_id is not None):
                # Fetch user object to get max_budget
                user_obj = await get_user_object(
                    user_id=user_api_key_dict.user_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    user_id_upsert=False,
                )

                if (
                    user_obj is not None 
                    and user_obj.max_budget is not None
                    and data.max_budget > user_obj.max_budget
                ):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": f"max budget higher than user max. User max budget={user_obj.max_budget}. User role={user_api_key_dict.user_role}"
                        },
                    )

            if data.models is not None and len(user_api_key_dict.models) > 0:
                for m in data.models:
                    if m not in user_api_key_dict.models:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": f"Model not in allowed user models. User allowed models={user_api_key_dict.models}. User id={user_api_key_dict.user_id}"
                            },
                        )

        if user_api_key_dict.user_id is not None:
            creating_user_in_list = False
            for member in data.members_with_roles:
                if member.user_id == user_api_key_dict.user_id:
                    creating_user_in_list = True

            if creating_user_in_list is False:
                data.members_with_roles.append(
                    Member(role="admin", user_id=user_api_key_dict.user_id)
                )

        ## ADD TO MODEL TABLE
        _model_id = None
        if data.model_aliases is not None and isinstance(data.model_aliases, dict):
            litellm_modeltable = LiteLLM_ModelTable(
                model_aliases=json.dumps(data.model_aliases),
                created_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
                updated_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
            )
            model_dict = await prisma_client.db.litellm_modeltable.create(
                {**litellm_modeltable.json(exclude_none=True)}  # type: ignore
            )  # type: ignore

            _model_id = model_dict.id

        ## Create Team Member Budget Table
        data_json = data.json()

        ## Handle Object Permission - MCP, Vector Stores etc.
        data_json = await _set_object_permission(
            data_json=data_json,
            prisma_client=prisma_client,
        )

        if TeamMemberBudgetHandler.should_create_budget(
            team_member_budget=data.team_member_budget,
            team_member_rpm_limit=data.team_member_rpm_limit,
            team_member_tpm_limit=data.team_member_tpm_limit,
        ):
            data_json = await TeamMemberBudgetHandler.create_team_member_budget_table(
                data=data,
                new_team_data_json=data_json,
                user_api_key_dict=user_api_key_dict,
                team_member_budget=data.team_member_budget,
                team_member_rpm_limit=data.team_member_rpm_limit,
                team_member_tpm_limit=data.team_member_tpm_limit,
            )

        ## ADD TO TEAM TABLE
        complete_team_data = LiteLLM_TeamTable(
            **data_json,
            model_id=_model_id,
        )

        # Set Management Endpoint Metadata Fields
        for field in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
            if getattr(data, field, None) is not None:
                _set_object_metadata_field(
                    object_data=complete_team_data,
                    field_name=field,
                    value=getattr(data, field),
                )

        for field in LiteLLM_ManagementEndpoint_MetadataFields:
            if getattr(data, field, None) is not None:
                _set_object_metadata_field(
                    object_data=complete_team_data,
                    field_name=field,
                    value=getattr(data, field),
                )

        # If budget_duration is set, set `budget_reset_at`
        if complete_team_data.budget_duration is not None:
            from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

            complete_team_data.budget_reset_at = get_budget_reset_time(
                budget_duration=complete_team_data.budget_duration,
            )

        ## Add Team Member Budget Table
        members_with_roles: List[Member] = []
        if complete_team_data.members_with_roles is not None:
            members_with_roles = complete_team_data.members_with_roles
            complete_team_data.members_with_roles = []

        complete_team_data_dict = complete_team_data.model_dump(exclude_none=True)
        complete_team_data_dict = prisma_client.jsonify_team_object(
            db_data=complete_team_data_dict
        )

        team_row: LiteLLM_TeamTable = await prisma_client.db.litellm_teamtable.create(
            data=complete_team_data_dict,
            include={"litellm_model_table": True},  # type: ignore
        )

        ## ADD TEAM ID TO USER TABLE ##
        team_member_add_request = TeamMemberAddRequest(
            team_id=data.team_id,
            member=members_with_roles,
        )
        await _add_team_members_to_team(
            data=team_member_add_request,
            complete_team_data=team_row,
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
        )

        # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
        if litellm.store_audit_logs is True:
            _updated_values = complete_team_data.json(exclude_none=True)

            _updated_values = json.dumps(_updated_values, default=str)

            asyncio.create_task(
                create_audit_log_for_update(
                    request_data=LiteLLM_AuditLogs(
                        id=str(uuid.uuid4()),
                        updated_at=datetime.now(timezone.utc),
                        changed_by=litellm_changed_by
                        or user_api_key_dict.user_id
                        or litellm_proxy_admin_name,
                        changed_by_api_key=user_api_key_dict.api_key,
                        table_name=LitellmTableNames.TEAM_TABLE_NAME,
                        object_id=data.team_id,
                        action="created",
                        updated_values=_updated_values,
                        before_value=None,
                    )
                )
            )

        try:
            return team_row.model_dump()
        except Exception:
            return team_row.dict()
    except Exception as e:
        raise handle_exception_on_proxy(e)


async def _create_team_update_audit_log(
    existing_team_row: LiteLLM_TeamTable,
    updated_kv: dict,
    team_id: str,
    litellm_changed_by: Optional[str],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
) -> None:
    """
    Create an audit log entry for team update operations.

    Args:
        existing_team_row: The team row before the update
        updated_kv: Dictionary of updated key-value pairs
        team_id: The ID of the team being updated
        litellm_changed_by: Optional header indicating who made the change
        user_api_key_dict: User API key authentication details
        litellm_proxy_admin_name: Name of the proxy admin
    """
    from litellm.proxy.management_helpers.audit_logs import create_audit_log_for_update

    _before_value = existing_team_row.json(exclude_none=True)
    _before_value = json.dumps(_before_value, default=str)
    _after_value: str = json.dumps(updated_kv, default=str)

    asyncio.create_task(
        create_audit_log_for_update(
            request_data=LiteLLM_AuditLogs(
                id=str(uuid.uuid4()),
                updated_at=datetime.now(timezone.utc),
                changed_by=litellm_changed_by
                or user_api_key_dict.user_id
                or litellm_proxy_admin_name,
                changed_by_api_key=user_api_key_dict.api_key,
                table_name=LitellmTableNames.TEAM_TABLE_NAME,
                object_id=team_id,
                action="updated",
                updated_values=_after_value,
                before_value=_before_value,
            )
        )
    )


async def _update_model_table(
    data: UpdateTeamRequest,
    model_id: Optional[str],
    prisma_client: PrismaClient,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
) -> Optional[str]:
    """
    Upsert model table and return the model id
    """
    ## UPSERT MODEL TABLE
    _model_id = model_id
    if data.model_aliases is not None and isinstance(data.model_aliases, dict):
        litellm_modeltable = LiteLLM_ModelTable(
            model_aliases=json.dumps(data.model_aliases),
            created_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
            updated_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
        )
        if model_id is None:
            model_dict = await prisma_client.db.litellm_modeltable.create(
                data={**litellm_modeltable.json(exclude_none=True)}  # type: ignore
            )
        else:
            model_dict = await prisma_client.db.litellm_modeltable.upsert(
                where={"id": model_id},
                data={
                    "update": {**litellm_modeltable.json(exclude_none=True)},  # type: ignore
                    "create": {**litellm_modeltable.json(exclude_none=True)},  # type: ignore
                },
            )  # type: ignore

        _model_id = model_dict.id

    return _model_id


async def fetch_and_validate_organization(
    organization_id: str,
    existing_team_row: Any,
    llm_router: Optional[Router],
    prisma_client: Any,
) -> Any:
    """
    Fetch and validate an organization for team update operations.

    Args:
        organization_id: The organization ID to fetch
        existing_team_row: The existing team row being updated
        llm_router: The LLM router instance
        prisma_client: The Prisma database client

    Returns:
        The organization row from the database

    Raises:
        HTTPException: If llm_router is None, organization not found, or validation fails
    """
    if llm_router is None:
        raise HTTPException(
            status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
        )

    organization_row = await prisma_client.db.litellm_organizationtable.find_unique(
        where={"organization_id": organization_id},
        include={"litellm_budget_table": True, "users": True},
    )

    if organization_row is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Organization not found, passed organization_id={organization_id}"
            },
        )

    validate_team_org_change(
        team=LiteLLM_TeamTable(**existing_team_row.model_dump()),
        organization=LiteLLM_OrganizationTable(**organization_row.model_dump()),
        llm_router=llm_router,
    )

    return organization_row


def validate_team_org_change(
    team: LiteLLM_TeamTable, organization: LiteLLM_OrganizationTable, llm_router: Router
) -> bool:
    """
    Validate that a team can be moved to an organization.

    - The org must have access to the team's models
    - The team budget cannot be greater than the org max_budget
    - The team's user_id must be a member of the org
    - The team's tpm/rpm limit must be less than the org's tpm/rpm limit
    """

    # If the team's organization is the same as the new organization, return True
    # Since no changes are being made
    if team.organization_id == organization.organization_id:
        return True

    # Check if the org has access to the team's models
    if len(organization.models) > 0:
        if SpecialModelNames.all_proxy_models.value in organization.models:
            pass
        elif team.models is None or len(team.models) == 0:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Cannot move team to organization. Team has access to all proxy models, but the organization does not."
                },
            )
        else:
            for model in team.models:
                can_org_access_model(
                    model=model,
                    org_object=organization,
                    llm_router=llm_router,
                )

    # Check if the team's budget is less than the org's max_budget
    if (
        team.max_budget
        and organization.litellm_budget_table
        and organization.litellm_budget_table.max_budget
        and team.max_budget > organization.litellm_budget_table.max_budget
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Cannot move team to organization. Team has max_budget {team.max_budget} that is greater than the organization's max_budget {organization.litellm_budget_table.max_budget}."
            },
        )

    # Check if the team's user_id is a member of the org
    team_members = [m.user_id for m in team.members_with_roles]
    org_members = [m.user_id for m in organization.users] if organization.users else []
    not_in_org = [
        m
        for m in team_members
        if m not in org_members and m != SpecialProxyStrings.default_user_id.value
    ]
    if len(not_in_org) > 0:
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Cannot move team to organization. Team has user_id {not_in_org} that is not a member of the organization."
            },
        )

    # Check if the team's tpm/rpm limit is less than the org's tpm/rpm limit
    if (
        team.tpm_limit
        and organization.litellm_budget_table
        and organization.litellm_budget_table.tpm_limit
        and team.tpm_limit > organization.litellm_budget_table.tpm_limit
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Cannot move team to organization. Team has tpm_limit {team.tpm_limit} that is greater than the organization's tpm_limit {organization.litellm_budget_table.tpm_limit}."
            },
        )
    if (
        team.rpm_limit
        and organization.litellm_budget_table
        and organization.litellm_budget_table.rpm_limit
        and team.rpm_limit > organization.litellm_budget_table.rpm_limit
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Cannot move team to organization. Team has rpm_limit {team.rpm_limit} that is greater than the organization's rpm_limit {organization.litellm_budget_table.rpm_limit}."
            },
        )
    return True


@router.post(
    "/team/update", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def update_team(
    data: UpdateTeamRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Use `/team/member_add` AND `/team/member/delete` to add/remove new team members

    You can now update team budget / rate limits via /team/update

    Parameters:
    - team_id: str - The team id of the user. Required param.
    - team_alias: Optional[str] - User defined team alias
    - team_member_permissions: Optional[List[str]] - A list of routes that non-admin team members can access. example: ["/key/generate", "/key/update", "/key/delete"]
    - metadata: Optional[dict] - Metadata for team, store information for team. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
    - tpm_limit: Optional[int] - The TPM (Tokens Per Minute) limit for this team - all keys with this team_id will have at max this TPM limit
    - rpm_limit: Optional[int] - The RPM (Requests Per Minute) limit for this team - all keys associated with this team_id will have at max this RPM limit
    - max_budget: Optional[float] - The maximum budget allocated to the team - all keys for this team_id will have at max this max_budget
    - budget_duration: Optional[str] - The duration of the budget for the team. Doc [here](https://docs.litellm.ai/docs/proxy/team_budgets)
    - models: Optional[list] - A list of models associated with the team - all keys for this team_id will have at most, these models. If empty, assumes all models are allowed.
    - prompts: Optional[List[str]] - List of prompts that the team is allowed to use.
    - blocked: bool - Flag indicating if the team is blocked or not - will stop all calls from keys with this team_id.
    - tags: Optional[List[str]] - Tags for [tracking spend](https://litellm.vercel.app/docs/proxy/enterprise#tracking-spend-for-custom-tags) and/or doing [tag-based routing](https://litellm.vercel.app/docs/proxy/tag_routing).
    - organization_id: Optional[str] - The organization id of the team. Default is None. Create via `/organization/new`.
    - model_aliases: Optional[dict] - Model aliases for the team. [Docs](https://docs.litellm.ai/docs/proxy/team_based_routing#create-team-with-model-alias)
    - guardrails: Optional[List[str]] - Guardrails for the team. [Docs](https://docs.litellm.ai/docs/proxy/guardrails)
    - prompts: Optional[List[str]] - List of prompts that the team is allowed to use.
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - team-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"], "mcp_tool_permissions": {"server_id_1": ["tool1", "tool2"]}}. IF null or {} then no object permission.
    - team_member_budget: Optional[float] - The maximum budget allocated to an individual team member.
    - team_member_rpm_limit: Optional[int] - The RPM (Requests Per Minute) limit for individual team members.
    - team_member_tpm_limit: Optional[int] - The TPM (Tokens Per Minute) limit for individual team members.
    - team_member_key_duration: Optional[str] - The duration for a team member's key. e.g. "1d", "1w", "1mo"
    - allowed_passthrough_routes: Optional[List[str]] - List of allowed pass through routes for the team.
    - model_rpm_limit: Optional[Dict[str, int]] - The RPM (Requests Per Minute) limit per model for this team. Example: {"gpt-4": 100, "gpt-3.5-turbo": 200}
    - model_tpm_limit: Optional[Dict[str, int]] - The TPM (Tokens Per Minute) limit per model for this team. Example: {"gpt-4": 10000, "gpt-3.5-turbo": 20000}
    Example - update team TPM Limit
    - allowed_vector_store_indexes: Optional[List[dict]] - List of allowed vector store indexes for the key. Example - [{"index_name": "my-index", "index_permissions": ["write", "read"]}]. If specified, the key will only be able to use these specific vector store indexes. Create index, using `/v1/indexes` endpoint.


    ```
    curl --location 'http://0.0.0.0:4000/team/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "team_id": "8d916b1c-510d-4894-a334-1c16a93344f5",
        "tpm_limit": 100
    }'
    ```

    Example - Update Team `max_budget` budget
    ```
    curl --location 'http://0.0.0.0:4000/team/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "team_id": "8d916b1c-510d-4894-a334-1c16a93344f5",
        "max_budget": 10
    }'
    ```
    """
    from litellm.proxy.auth.auth_checks import _cache_team_object
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        llm_router,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if data.team_id is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})
    verbose_proxy_logger.debug("/team/update - %s", data)

    existing_team_row = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": data.team_id}
    )

    if existing_team_row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Team not found, passed team_id={data.team_id}"},
        )

    if (
        data.organization_id is not None and len(data.organization_id) > 0
    ):  # allow unsetting the organization_id
        await fetch_and_validate_organization(
            organization_id=data.organization_id,
            existing_team_row=existing_team_row,
            llm_router=llm_router,
            prisma_client=prisma_client,
        )
    elif data.organization_id is not None and len(data.organization_id) == 0:
        # unsetting the organization_id
        data.organization_id = None

    # check org team limits - if updating team that belongs to an org
    org_id_to_check = (
        data.organization_id
        if data.organization_id is not None
        else existing_team_row.organization_id
    )
    if (
        org_id_to_check is not None
        and isinstance(org_id_to_check, str)
        and prisma_client is not None
    ):
        org_table = await get_org_object(
            org_id=org_id_to_check,
            user_api_key_cache=user_api_key_cache,
            prisma_client=prisma_client,
        )
        if org_table is not None:
            await _check_org_team_limits(
                org_table=org_table,
                data=data,
                prisma_client=prisma_client,
            )

    updated_kv = data.json(exclude_unset=True)

    # Check budget_duration and budget_reset_at
    if data.budget_duration is not None:
        from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

        reset_at = get_budget_reset_time(budget_duration=data.budget_duration)

        # set the budget_reset_at in DB
        updated_kv["budget_reset_at"] = reset_at

    if TeamMemberBudgetHandler.should_create_budget(
        team_member_budget=data.team_member_budget,
        team_member_rpm_limit=data.team_member_rpm_limit,
        team_member_tpm_limit=data.team_member_tpm_limit,
    ):
        updated_kv = await TeamMemberBudgetHandler.upsert_team_member_budget_table(
            team_table=existing_team_row,
            user_api_key_dict=user_api_key_dict,
            updated_kv=updated_kv,
            team_member_budget=data.team_member_budget,
            team_member_rpm_limit=data.team_member_rpm_limit,
            team_member_tpm_limit=data.team_member_tpm_limit,
        )
    else:
        TeamMemberBudgetHandler._clean_team_member_fields(updated_kv)

    # Check object permission
    if data.object_permission is not None:
        updated_kv = await handle_update_object_permission(
            data_json=updated_kv,
            existing_team_row=existing_team_row,
        )

    # update team metadata fields
    _team_metadata_fields = LiteLLM_ManagementEndpoint_MetadataFields_Premium
    for field in _team_metadata_fields:
        if field in updated_kv and updated_kv[field] is not None:
            _update_metadata_field(
                updated_kv=updated_kv,
                field_name=field,
            )

    for field in LiteLLM_ManagementEndpoint_MetadataFields:
        if field in updated_kv and updated_kv[field] is not None:
            _update_metadata_field(
                updated_kv=updated_kv,
                field_name=field,
            )

    if "model_aliases" in updated_kv:
        updated_kv.pop("model_aliases")
        _model_id = await _update_model_table(
            data=data,
            model_id=existing_team_row.model_id,
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
        )
        if _model_id is not None:
            updated_kv["model_id"] = _model_id

    updated_kv = prisma_client.jsonify_team_object(db_data=updated_kv)
    team_row: Optional[LiteLLM_TeamTable] = (
        await prisma_client.db.litellm_teamtable.update(
            where={"team_id": data.team_id},
            data=updated_kv,
            include={"litellm_model_table": True},  # type: ignore
        )
    )

    if team_row is None or team_row.team_id is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Team doesn't exist. Got={}".format(team_row)},
        )

    verbose_proxy_logger.info("Successfully updated team - %s, info", team_row.team_id)
    await _cache_team_object(
        team_id=team_row.team_id,
        team_table=LiteLLM_TeamTableCachedObj(**team_row.model_dump()),
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
    if litellm.store_audit_logs is True:
        await _create_team_update_audit_log(
            existing_team_row=existing_team_row,
            updated_kv=updated_kv,
            team_id=data.team_id,
            litellm_changed_by=litellm_changed_by,
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
        )

    return {"team_id": team_row.team_id, "data": team_row}


async def handle_update_object_permission(
    data_json: dict, existing_team_row: LiteLLM_TeamTable
) -> dict:
    """
    Handle the update of object permission for a team.

    - IF there's no object_permission_id, then create a new entry in LiteLLM_ObjectPermissionTable
    - IF there's an object_permission_id, then update the entry in LiteLLM_ObjectPermissionTable
    """
    from litellm.proxy.proxy_server import prisma_client

    # Use the common helper to handle the object permission update
    object_permission_id = await handle_update_object_permission_common(
        data_json=data_json,
        existing_object_permission_id=existing_team_row.object_permission_id,
        prisma_client=prisma_client,
    )

    # Add the object_permission_id to data_json if one was created/updated
    if object_permission_id is not None:
        data_json["object_permission_id"] = object_permission_id
        verbose_proxy_logger.debug(
            f"updated object_permission_id: {object_permission_id}"
        )

    return data_json


def _check_team_member_admin_add(
    member: Union[Member, List[Member]],
    premium_user: bool,
):
    if isinstance(member, Member) and member.role == "admin":
        if premium_user is not True:
            raise ValueError(
                f"Assigning team admins is a premium feature. {CommonProxyErrors.not_premium_user.value}"
            )
    elif isinstance(member, List):
        for m in member:
            if m.role == "admin":
                if premium_user is not True:
                    raise ValueError(
                        f"Assigning team admins is a premium feature. Got={m}. {CommonProxyErrors.not_premium_user.value}. "
                    )


def team_call_validation_checks(
    prisma_client: Optional[PrismaClient],
    data: TeamMemberAddRequest,
    premium_user: bool,
):
    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_id is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    if data.member is None:
        raise HTTPException(
            status_code=400, detail={"error": "No member/members passed in"}
        )

    try:
        _check_team_member_admin_add(
            member=data.member,
            premium_user=premium_user,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})


def team_member_add_duplication_check(
    data: TeamMemberAddRequest,
    existing_team_row: LiteLLM_TeamTable,
):
    """
    Check if a member already exists in the team.
    This check is done BEFORE we create/fetch the user, so it only prevents
    obvious duplicates where both user_id and user_email match exactly.
    """

    invalid_team_members = []

    def _check_member_duplication(member: Member):
        if member.user_id is not None:
            for existing_member in existing_team_row.members_with_roles:
                if existing_member.user_id == member.user_id:
                    invalid_team_members.append(member)

        # Check by user_email if provided
        if member.user_email is not None:
            for existing_member in existing_team_row.members_with_roles:
                if existing_member.user_email == member.user_email:
                    invalid_team_members.append(member)

    # First, populate the invalid_team_members list by checking for duplicates
    if isinstance(data.member, Member):
        _check_member_duplication(data.member)
    elif isinstance(data.member, List):
        for m in data.member:
            _check_member_duplication(m)

    # Then check the populated list and raise exceptions if needed
    if isinstance(data.member, list) and len(invalid_team_members) == len(data.member):
        raise ProxyException(
            message=f"All users are already in team. Existing members={existing_team_row.members_with_roles}",
            type=ProxyErrorTypes.team_member_already_in_team,
            param="member",
            code="400",
        )
    elif isinstance(data.member, Member) and len(invalid_team_members) == 1:
        raise ProxyException(
            message=f"User already in team. Member: user_id={data.member.user_id}, user_email={data.member.user_email}. Existing members={existing_team_row.members_with_roles}",
            type=ProxyErrorTypes.team_member_already_in_team,
            param="member",
            code="400",
        )
    elif len(invalid_team_members) > 0:
        verbose_proxy_logger.info(
            f"Some users are already in team. Existing members={existing_team_row.members_with_roles}. Duplicate members={invalid_team_members}",
        )


async def _validate_team_member_add_permissions(
    user_api_key_dict: UserAPIKeyAuth,
    complete_team_data: LiteLLM_TeamTable,
) -> None:
    """Validate if user has permission to add members to the team."""
    if (
        hasattr(user_api_key_dict, "user_role")
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(
            user_api_key_dict=user_api_key_dict, team_obj=complete_team_data
        )
        and not _is_available_team(
            team_id=complete_team_data.team_id,
            user_api_key_dict=user_api_key_dict,
        )
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Call not allowed. User not proxy admin OR team admin. route={}, team_id={}".format(
                    "/team/member_add",
                    complete_team_data.team_id,
                )
            },
        )


async def _process_team_members(
    data: TeamMemberAddRequest,
    complete_team_data: LiteLLM_TeamTable,
    prisma_client: PrismaClient,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
) -> Tuple[List[LiteLLM_UserTable], List[LiteLLM_TeamMembership]]:
    """Process and add new team members."""
    updated_users: List[LiteLLM_UserTable] = []
    updated_team_memberships: List[LiteLLM_TeamMembership] = []

    default_team_budget_id = (
        complete_team_data.metadata.get("team_member_budget_id")
        if complete_team_data.metadata is not None
        else None
    )

    if isinstance(data.member, Member):
        try:
            updated_user, updated_tm = await add_new_member(
                new_member=data.member,
                max_budget_in_team=data.max_budget_in_team,
                prisma_client=prisma_client,
                user_api_key_dict=user_api_key_dict,
                litellm_proxy_admin_name=litellm_proxy_admin_name,
                team_id=data.team_id,
                default_team_budget_id=default_team_budget_id,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Unable to add user - {}, to team - {}, for reason - {}".format(
                        data.member, data.team_id, str(e)
                    )
                },
            )
        updated_users.append(updated_user)
        if updated_tm is not None:
            updated_team_memberships.append(updated_tm)
    elif isinstance(data.member, List):
        for m in data.member:
            try:
                updated_user, updated_tm = await add_new_member(
                    new_member=m,
                    max_budget_in_team=data.max_budget_in_team,
                    prisma_client=prisma_client,
                    user_api_key_dict=user_api_key_dict,
                    litellm_proxy_admin_name=litellm_proxy_admin_name,
                    team_id=data.team_id,
                    default_team_budget_id=default_team_budget_id,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Unable to add user - {}, to team - {}, for reason - {}".format(
                            m, data.team_id, str(e)
                        )
                    },
                )
            updated_users.append(updated_user)
            if updated_tm is not None:
                updated_team_memberships.append(updated_tm)

    return updated_users, updated_team_memberships


async def _update_team_members_list(
    data: TeamMemberAddRequest,
    complete_team_data: LiteLLM_TeamTable,
    updated_users: List[LiteLLM_UserTable],
) -> None:
    """Update the team's members_with_roles list."""
    if isinstance(data.member, Member):
        new_member = data.member.model_copy()

        # get user id
        if new_member.user_id is None and new_member.user_email is not None:
            for user in updated_users:
                if (
                    user.user_email is not None
                    and user.user_email == new_member.user_email
                ):
                    new_member.user_id = user.user_id

        # Check if member already exists in team before adding
        member_already_exists = False
        for existing_member in complete_team_data.members_with_roles:
            if (
                new_member.user_id is not None
                and existing_member.user_id == new_member.user_id
            ) or (
                new_member.user_email is not None
                and existing_member.user_email == new_member.user_email
            ):
                member_already_exists = True
                break

        if not member_already_exists:
            complete_team_data.members_with_roles.append(new_member)

    elif isinstance(data.member, List):
        for nm in data.member:
            if nm.user_id is None and nm.user_email is not None:
                for user in updated_users:
                    if user.user_email is not None and user.user_email == nm.user_email:
                        nm.user_id = user.user_id

            # Check if member already exists in team before adding
            member_already_exists = False
            for existing_member in complete_team_data.members_with_roles:
                if (
                    nm.user_id is not None and existing_member.user_id == nm.user_id
                ) or (
                    nm.user_email is not None
                    and existing_member.user_email == nm.user_email
                ):
                    member_already_exists = True
                    break

            if not member_already_exists:
                complete_team_data.members_with_roles.append(nm)


async def _add_team_members_to_team(
    data: TeamMemberAddRequest,
    complete_team_data: LiteLLM_TeamTable,
    prisma_client: PrismaClient,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
) -> Tuple[LiteLLM_TeamTable, List[LiteLLM_UserTable], List[LiteLLM_TeamMembership]]:
    """Add team members to the team."""
    # Process and add new members
    updated_users, updated_team_memberships = await _process_team_members(
        data=data,
        complete_team_data=complete_team_data,
        prisma_client=prisma_client,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
    )

    # Update team members list
    await _update_team_members_list(
        data=data,
        complete_team_data=complete_team_data,
        updated_users=updated_users,
    )

    # ADD MEMBER TO TEAM
    _db_team_members = [m.model_dump() for m in complete_team_data.members_with_roles]
    updated_team = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id},
        data={"members_with_roles": json.dumps(_db_team_members)},  # type: ignore
    )

    return updated_team, updated_users, updated_team_memberships


@router.post(
    "/team/member_add",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=TeamAddMemberResponse,
)
@management_endpoint_wrapper
async def team_member_add(
    data: TeamMemberAddRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Add new members (either via user_email or user_id) to a team

    If user doesn't exist, new user row will also be added to User Table

    Only proxy_admin or admin of team, allowed to access this endpoint.
    ```

    curl -X POST 'http://0.0.0.0:4000/team/member_add' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -d '{"team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849", "member": {"role": "user", "user_id": "krrish247652@berri.ai"}}'

    ```
    """
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        premium_user,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    try:
        team_call_validation_checks(
            prisma_client=prisma_client,
            data=data,
            premium_user=premium_user,
        )
    except HTTPException as e:
        raise e

    prisma_client = cast(PrismaClient, prisma_client)

    existing_team_row = await get_team_object(
        team_id=data.team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=proxy_logging_obj,
        check_cache_only=False,
        check_db_only=True,
    )
    if existing_team_row is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Team not found for team_id={getattr(data, 'team_id', None)}"
            },
        )

    complete_team_data = LiteLLM_TeamTable(**existing_team_row.model_dump())

    team_member_add_duplication_check(
        data=data,
        existing_team_row=complete_team_data,
    )

    # Validate permissions
    await _validate_team_member_add_permissions(
        user_api_key_dict=user_api_key_dict,
        complete_team_data=complete_team_data,
    )

    updated_team, updated_users, updated_team_memberships = (
        await _add_team_members_to_team(
            data=data,
            complete_team_data=complete_team_data,
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
        )
    )

    # Check if updated_team is None
    if updated_team is None:
        raise HTTPException(
            status_code=404, detail={"error": f"Team with id {data.team_id} not found"}
        )
    return TeamAddMemberResponse(
        **updated_team.model_dump(),
        updated_users=updated_users,
        updated_team_memberships=updated_team_memberships,
    )


def _cleanup_members_with_roles(
    existing_team_row: LiteLLM_TeamTable,
    data: TeamMemberDeleteRequest,
) -> Tuple[bool, List[Member]]:
    """Cleanup members_with_roles list for a team."""
    is_member_in_team = False
    new_team_members: List[Member] = []
    for m in existing_team_row.members_with_roles:
        if (
            data.user_id is not None
            and m.user_id is not None
            and data.user_id == m.user_id
        ):
            is_member_in_team = True
            continue
        elif (
            data.user_email is not None
            and m.user_email is not None
            and data.user_email == m.user_email
        ):
            is_member_in_team = True
            continue
        new_team_members.append(m)
    return is_member_in_team, new_team_members


@router.post(
    "/team/member_delete",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def team_member_delete(
    data: TeamMemberDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA]

    delete members (either via user_email or user_id) from a team

    If user doesn't exist, an exception will be raised
    ```
    curl -X POST 'http://0.0.0.0:8000/team/member_delete' \

    -H 'Authorization: Bearer sk-1234' \

    -H 'Content-Type: application/json' \

    -d '{
        "team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849",
        "user_id": "krrish247652@berri.ai"
    }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_id is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    if data.user_id is None and data.user_email is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Either user_id or user_email needs to be passed in"},
        )

    _existing_team_row = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": data.team_id}
    )

    if _existing_team_row is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Team id={} does not exist in db".format(data.team_id)},
        )
    existing_team_row = LiteLLM_TeamTable(**_existing_team_row.model_dump())

    ## CHECK IF USER IS PROXY ADMIN OR TEAM ADMIN

    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(
            user_api_key_dict=user_api_key_dict, team_obj=existing_team_row
        )
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Call not allowed. User not proxy admin OR team admin. route={}, team_id={}".format(
                    "/team/member_delete", existing_team_row.team_id
                )
            },
        )

    ## DELETE MEMBER FROM TEAM
    is_member_in_team, new_team_members = _cleanup_members_with_roles(
        existing_team_row=existing_team_row,
        data=data,
    )

    if not is_member_in_team:
        raise HTTPException(status_code=400, detail={"error": "User not found in team"})

    existing_team_row.members_with_roles = new_team_members

    _db_new_team_members: List[dict] = [m.model_dump() for m in new_team_members]

    _ = await prisma_client.db.litellm_teamtable.update(
        where={
            "team_id": data.team_id,
        },
        data={"members_with_roles": json.dumps(_db_new_team_members)},  # type: ignore
    )

    ## DELETE TEAM ID from USER ROW, IF EXISTS ##
    # get user row
    key_val = {}
    if data.user_id is not None:
        key_val["user_id"] = data.user_id
    elif data.user_email is not None:
        key_val["user_email"] = data.user_email
    existing_user_rows = await prisma_client.db.litellm_usertable.find_many(
        where=key_val  # type: ignore
    )

    if existing_user_rows is not None and (
        isinstance(existing_user_rows, list) and len(existing_user_rows) > 0
    ):
        for existing_user in existing_user_rows:
            team_list = []
            if data.team_id in existing_user.teams:
                team_list = existing_user.teams
                team_list.remove(data.team_id)
                await prisma_client.db.litellm_usertable.update(
                    where={
                        "user_id": existing_user.user_id,
                    },
                    data={"teams": {"set": team_list}},
                )

    # Also clean up any existing team membership rows for this user and team
    user_ids_to_delete = set()
    if data.user_id is not None:
        user_ids_to_delete.add(data.user_id)
    if existing_user_rows is not None and isinstance(existing_user_rows, list):
        for existing_user in existing_user_rows:
            if getattr(existing_user, "user_id", None):
                user_ids_to_delete.add(existing_user.user_id)

    for _uid in user_ids_to_delete:
        await prisma_client.db.litellm_teammembership.delete_many(
            where={"team_id": data.team_id, "user_id": _uid}
        )

    return existing_team_row


@router.post(
    "/team/member_update",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=TeamMemberUpdateResponse,
)
@management_endpoint_wrapper
async def team_member_update(
    data: TeamMemberUpdateRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA]

    Update team member budgets and team member role
    """
    from litellm.proxy.proxy_server import premium_user, prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_id is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    if data.role == "admin" and not premium_user:
        # exactly the same text your proxy throws for add:
        raise HTTPException(
            status_code=400,
            detail="Assigning team admins is a premium feature. You must be a LiteLLM Enterprise user to use this feature. If you have a license please set `LITELLM_LICENSE` in your env. Get a 7 day trial key here: https://www.litellm.ai/#trial. Pricing: https://www.litellm.ai/#pricing",
        )
    if data.user_id is None and data.user_email is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Either user_id or user_email needs to be passed in"},
        )

    _existing_team_row = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": data.team_id}
    )

    if _existing_team_row is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Team id={} does not exist in db".format(data.team_id)},
        )
    existing_team_row = LiteLLM_TeamTable(**_existing_team_row.model_dump())

    ## CHECK IF USER IS PROXY ADMIN OR TEAM ADMIN

    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(
            user_api_key_dict=user_api_key_dict, team_obj=existing_team_row
        )
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Call not allowed. User not proxy admin OR team admin. route={}, team_id={}".format(
                    "/team/member_delete", existing_team_row.team_id
                )
            },
        )

    returned_team_info: TeamInfoResponseObject = await team_info(
        http_request=http_request,
        team_id=data.team_id,
        user_api_key_dict=user_api_key_dict,
    )

    team_table = returned_team_info["team_info"]

    ## get user id
    received_user_id: Optional[str] = None
    if data.user_id is not None:
        received_user_id = data.user_id
    elif data.user_email is not None:
        for member in returned_team_info["team_info"].members_with_roles:
            if member.user_email is not None and member.user_email == data.user_email:
                received_user_id = member.user_id
                break

    if received_user_id is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "User id doesn't exist in team table. Data={}".format(data)
            },
        )
    ## find the relevant team membership
    identified_budget_id: Optional[str] = None
    for tm in returned_team_info["team_memberships"]:
        if tm.user_id == received_user_id:
            identified_budget_id = tm.budget_id
            break

    ### upsert new budget
    async with prisma_client.db.tx() as tx:
        await _upsert_budget_and_membership(
            tx=tx,
            team_id=data.team_id,
            user_id=received_user_id,
            max_budget=data.max_budget_in_team,
            existing_budget_id=identified_budget_id,
            user_api_key_dict=user_api_key_dict,
            tpm_limit=data.tpm_limit,
            rpm_limit=data.rpm_limit,
        )

    ### update team member role
    if data.role is not None:
        team_members: List[Member] = []
        for member in team_table.members_with_roles:
            if member.user_id == received_user_id:
                team_members.append(
                    Member(
                        user_id=member.user_id,
                        role=data.role,
                        user_email=data.user_email or member.user_email,
                    )
                )
            else:
                team_members.append(member)

        team_table.members_with_roles = team_members

        _db_team_members: List[dict] = [m.model_dump() for m in team_members]
        await prisma_client.db.litellm_teamtable.update(
            where={"team_id": data.team_id},
            data={"members_with_roles": json.dumps(_db_team_members)},  # type: ignore
        )

    return TeamMemberUpdateResponse(
        team_id=data.team_id,
        user_id=received_user_id,
        user_email=data.user_email,
        max_budget_in_team=data.max_budget_in_team,
        tpm_limit=data.tpm_limit,
        rpm_limit=data.rpm_limit,
    )


def _create_results_from_response(
    members: List[Member],
    response: TeamAddMemberResponse,
) -> List[TeamMemberAddResult]:
    """
    Convert TeamAddMemberResponse into individual TeamMemberAddResult objects
    """
    results: List[TeamMemberAddResult] = []

    for member in members:
        # Find corresponding updated user
        updated_user = None
        for user in response.updated_users:
            if (member.user_id and user.user_id == member.user_id) or (
                member.user_email and user.user_email == member.user_email
            ):
                updated_user = user.model_dump()
                break

        # Find corresponding updated team membership
        updated_team_membership = None
        for tm in response.updated_team_memberships:
            if member.user_id and tm.user_id == member.user_id:
                updated_team_membership = tm.model_dump()
                break

        results.append(
            TeamMemberAddResult(
                user_id=member.user_id,
                user_email=member.user_email,
                success=True,
                updated_user=updated_user,
                updated_team_membership=updated_team_membership,
            )
        )

    return results


@router.post(
    "/team/bulk_member_add",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=BulkTeamMemberAddResponse,
)
@management_endpoint_wrapper
async def bulk_team_member_add(
    data: BulkTeamMemberAddRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Bulk add multiple members to a team at once.
    
    This endpoint reuses the same logic as /team/member_add but provides a bulk-friendly response format.
    
    Parameters:
    - team_id: str - The ID of the team to add members to
    - members: List[Member] - List of members to add to the team
    - all_users: Optional[bool] - Flag to add all users on Proxy to the team
    - max_budget_in_team: Optional[float] - Maximum budget allocated to each user within the team
    
    Returns:
    - results: List of individual member addition results
    - total_requested: Total number of members requested for addition
    - successful_additions: Number of successful additions  
    - failed_additions: Number of failed additions
    - updated_team: The updated team object
    
    Example request:
    ```bash
    curl --location 'http://0.0.0.0:4000/team/bulk_member_add' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "team_id": "team-1234",
        "members": [
            {
                "user_id": "user1",
                "role": "user"
            },
            {
                "user_email": "user2@example.com",
                "role": "admin"
            }
        ],
        "max_budget_in_team": 100.0
    }'
    ```
    """
    from litellm.proxy._types import CommonProxyErrors
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if data.all_users:
        # get all users from the database
        all_users_in_db = await prisma_client.db.litellm_usertable.find_many(
            order={"created_at": "desc"}
        )
        data.members = [
            Member(
                user_id=user.user_id,
                user_email=user.user_email,
                role="user",
            )
            for user in all_users_in_db
        ]

    if not data.members:
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one member is required"},
        )

    # Limit batch size to prevent overwhelming the system
    MAX_BATCH_SIZE = 500
    if len(data.members) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Maximum {MAX_BATCH_SIZE} members can be added at once"},
        )

    try:
        # Reuse the existing team_member_add logic directly
        response = await team_member_add(
            data=TeamMemberAddRequest(
                team_id=data.team_id,
                member=data.members,  # Pass the entire list
                max_budget_in_team=data.max_budget_in_team,
            ),
            user_api_key_dict=user_api_key_dict,
        )

        # Convert to bulk response format
        results = _create_results_from_response(data.members, response)

        return BulkTeamMemberAddResponse(
            team_id=data.team_id,
            results=results,
            total_requested=len(data.members),
            successful_additions=len(results),  # All succeeded if we got here
            failed_additions=0,
            updated_team=response.model_dump(),
        )

    except Exception as e:
        # If the entire operation fails, mark all members as failed
        verbose_proxy_logger.exception(e)
        error_message = str(e)
        results = [
            TeamMemberAddResult(
                user_id=member.user_id,
                user_email=member.user_email,
                success=False,
                error=error_message,
            )
            for member in data.members
        ]

        return BulkTeamMemberAddResponse(
            team_id=data.team_id,
            results=results,
            total_requested=len(data.members),
            successful_additions=0,
            failed_additions=len(data.members),
            updated_team=None,
        )


@router.post(
    "/team/delete", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def delete_team(
    data: DeleteTeamRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    delete team and associated team keys

    Parameters:
    - team_ids: List[str] - Required. List of team IDs to delete. Example: ["team-1234", "team-5678"]

    ```
    curl --location 'http://0.0.0.0:4000/team/delete' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "team_ids": ["8d916b1c-510d-4894-a334-1c16a93344f5"]
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_ids is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    # check that all teams passed exist
    team_rows: List[LiteLLM_TeamTable] = []
    for team_id in data.team_ids:
        try:
            team_row_base: Optional[BaseModel] = (
                await prisma_client.db.litellm_teamtable.find_unique(
                    where={"team_id": team_id}
                )
            )
            if team_row_base is None:
                raise Exception
        except Exception:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team not found, passed team_id={team_id}"},
            )
        team_row_pydantic = LiteLLM_TeamTable(**team_row_base.model_dump())
        team_rows.append(team_row_pydantic)

    # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
    # we do this after the first for loop, since first for loop is for validation. we only want this inserted after validation passes
    if litellm.store_audit_logs is True:
        # make an audit log for each team deleted
        for team_id in data.team_ids:
            team_row: Optional[LiteLLM_TeamTable] = await prisma_client.get_data(  # type: ignore
                team_id=team_id, table_name="team", query_type="find_unique"
            )

            if team_row is None:
                continue

            _team_row = team_row.json(exclude_none=True)

            asyncio.create_task(
                create_audit_log_for_update(
                    request_data=LiteLLM_AuditLogs(
                        id=str(uuid.uuid4()),
                        updated_at=datetime.now(timezone.utc),
                        changed_by=litellm_changed_by
                        or user_api_key_dict.user_id
                        or litellm_proxy_admin_name,
                        changed_by_api_key=user_api_key_dict.api_key,
                        table_name=LitellmTableNames.TEAM_TABLE_NAME,
                        object_id=team_id,
                        action="deleted",
                        updated_values="{}",
                        before_value=_team_row,
                    )
                )
            )

    # End of Audit logging

    ## DELETE ASSOCIATED KEYS
    await prisma_client.delete_data(team_id_list=data.team_ids, table_name="key")

    # ## DELETE TEAM MEMBERSHIPS
    for team_row in team_rows:
        ### get all team members
        team_members = team_row.members_with_roles
        ### call team_member_delete for each team member
        tasks = []
        for team_member in team_members:
            tasks.append(
                team_member_delete(
                    data=TeamMemberDeleteRequest(
                        team_id=team_row.team_id,
                        user_id=team_member.user_id,
                        user_email=team_member.user_email,
                    ),
                    user_api_key_dict=user_api_key_dict,
                )
            )
        await asyncio.gather(*tasks)

    ## DELETE TEAMS
    deleted_teams = await prisma_client.delete_data(
        team_id_list=data.team_ids, table_name="team"
    )
    return deleted_teams


def validate_membership(
    user_api_key_dict: UserAPIKeyAuth, team_table: LiteLLM_TeamTable
):
    if (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value
    ):
        return

    if (
        user_api_key_dict.team_id == team_table.team_id
    ):  # allow team keys to check their info
        return

    if user_api_key_dict.user_id not in [
        m.user_id for m in team_table.members_with_roles
    ]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "User={} not authorized to access this team={}".format(
                    user_api_key_dict.user_id, team_table.team_id
                )
            },
        )


def _unfurl_all_proxy_models(
    team_info: LiteLLM_TeamTable, llm_router: Router
) -> LiteLLM_TeamTable:
    if (
        SpecialModelNames.all_proxy_models.value in team_info.models
        and llm_router is not None
    ):
        team_models: set[str] = set()  # make set to avoid duplicates
        for model in team_info.models:
            if model != SpecialModelNames.all_proxy_models.value:
                team_models.add(model)
        for model in llm_router.get_model_names():
            team_models.add(model)
        team_info.models = list(team_models)
    return team_info


async def _add_team_member_budget_table(
    team_member_budget_id: str,
    prisma_client: PrismaClient,
    team_info_response_object: TeamInfoResponseObjectTeamTable,
) -> TeamInfoResponseObjectTeamTable:
    try:
        team_budget = await prisma_client.db.litellm_budgettable.find_unique(
            where={"budget_id": team_member_budget_id}
        )
        team_info_response_object.team_member_budget_table = team_budget
    except Exception:
        verbose_proxy_logger.info(
            f"Team member budget table not found, passed team_member_budget_id={team_member_budget_id}"
        )

    return team_info_response_object


@router.get(
    "/team/info", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def team_info(
    http_request: Request,
    team_id: str = fastapi.Query(
        default=None, description="Team ID in the request parameters"
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    get info on team + related keys

    Parameters:
    - team_id: str - Required. The unique identifier of the team to get info on.

    ```
    curl --location 'http://localhost:4000/team/info?team_id=your_team_id_here' \
    --header 'Authorization: Bearer your_api_key_here'
    ```
    """
    from litellm.proxy._types import TeamInfoResponseObjectTeamTable
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
                },
            )
        if team_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Malformed request. No team id passed in."},
            )

        try:
            team_info: Optional[BaseModel] = (
                await prisma_client.db.litellm_teamtable.find_unique(
                    where={"team_id": team_id},
                    include={"object_permission": True},
                )
            )
            if team_info is None:
                raise Exception
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": f"Team not found, passed team id: {team_id}."},
            )
        validate_membership(
            user_api_key_dict=user_api_key_dict,
            team_table=LiteLLM_TeamTable(**team_info.model_dump()),
        )

        ## GET ALL KEYS ##
        keys = await prisma_client.get_data(
            team_id=team_id,
            table_name="key",
            query_type="find_all",
            expires=datetime.now(),
        )

        if keys is None:
            keys = []

        if team_info is None:
            ## make sure we still return a total spend ##
            spend = 0
            for k in keys:
                spend += getattr(k, "spend", 0)
            team_info = {"spend": spend}

        ## REMOVE HASHED TOKEN INFO before returning ##
        for key in keys:
            try:
                key = key.model_dump()  # noqa
            except Exception:
                # if using pydantic v1
                key = key.dict()
            key.pop("token", None)

        ## GET ALL MEMBERSHIPS ##
        returned_tm = await get_all_team_memberships(
            prisma_client, [team_id], user_id=None
        )

        if isinstance(team_info, dict):
            _team_info = TeamInfoResponseObjectTeamTable(**team_info)
        elif isinstance(team_info, BaseModel):
            _team_info = TeamInfoResponseObjectTeamTable(**team_info.model_dump())
        else:
            _team_info = TeamInfoResponseObjectTeamTable()

        ## GET TEAM BUDGET (if exists) ##
        team_member_budget_id = (
            _team_info.metadata.get("team_member_budget_id")
            if _team_info.metadata is not None
            else None
        )
        if team_member_budget_id is not None:
            _team_info = await _add_team_member_budget_table(
                team_member_budget_id=team_member_budget_id,
                prisma_client=prisma_client,
                team_info_response_object=_team_info,
            )

        # ## UNFURL 'all-proxy-models' into the team_info.models list ##
        # if llm_router is not None:
        #     _team_info = _unfurl_all_proxy_models(_team_info, llm_router)
        response_object = TeamInfoResponseObject(
            team_id=team_id,
            team_info=_team_info,
            keys=keys,
            team_memberships=returned_tm,
        )
        return response_object

    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.management_endpoints.team_endpoints.py::team_info - Exception occurred - {}\n{}".format(
                e, traceback.format_exc()
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
    "/team/block", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def block_team(
    data: BlockTeamRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Blocks all calls from keys with this team id.

    Parameters:
    - team_id: str - Required. The unique identifier of the team to block.

    Example:
    ```
    curl --location 'http://0.0.0.0:4000/team/block' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "team_id": "team-1234"
    }'
    ```

    Returns:
    - The updated team record with blocked=True



    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise Exception("No DB Connected.")

    record = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id}, data={"blocked": True}  # type: ignore
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Team not found, passed team_id={data.team_id}"},
        )

    return record


@router.post(
    "/team/unblock", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def unblock_team(
    data: BlockTeamRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Blocks all calls from keys with this team id.

    Parameters:
    - team_id: str - Required. The unique identifier of the team to unblock.

    Example:
    ```
    curl --location 'http://0.0.0.0:4000/team/unblock' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "team_id": "team-1234"
    }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise Exception("No DB Connected.")

    record = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id}, data={"blocked": False}  # type: ignore
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Team not found, passed team_id={data.team_id}"},
        )

    return record


@router.get("/team/available")
async def list_available_teams(
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    response_model=List[LiteLLM_TeamTable],
):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    available_teams = cast(
        Optional[List[str]],
        (
            litellm.default_internal_user_params.get("available_teams")
            if litellm.default_internal_user_params is not None
            else None
        ),
    )
    if available_teams is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No available teams for user to join. See how to set available teams here: https://docs.litellm.ai/docs/proxy/self_serve#all-settings-for-self-serve--sso-flow"
            },
        )

    # filter out teams that the user is already a member of
    user_info = await prisma_client.db.litellm_usertable.find_unique(
        where={"user_id": user_api_key_dict.user_id}
    )
    if user_info is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "User not found"},
        )
    user_info_correct_type = LiteLLM_UserTable(**user_info.model_dump())

    available_teams = [
        team for team in available_teams if team not in user_info_correct_type.teams
    ]

    available_teams_db = await prisma_client.db.litellm_teamtable.find_many(
        where={"team_id": {"in": available_teams}}
    )

    available_teams_correct_type = [
        LiteLLM_TeamTable(**team.model_dump()) for team in available_teams_db
    ]

    return available_teams_correct_type


@router.get(
    "/v2/team/list",
    tags=["team management"],
    response_model=TeamListResponse,
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def list_team_v2(
    http_request: Request,
    user_id: Optional[str] = fastapi.Query(
        default=None, description="Only return teams which this 'user_id' belongs to"
    ),
    organization_id: Optional[str] = fastapi.Query(
        default=None,
        description="Only return teams which this 'organization_id' belongs to",
    ),
    team_id: Optional[str] = fastapi.Query(
        default=None, description="Only return teams which this 'team_id' belongs to"
    ),
    team_alias: Optional[str] = fastapi.Query(
        default=None,
        description="Only return teams which this 'team_alias' belongs to. Supports partial matching.",
    ),
    page: int = fastapi.Query(
        default=1, description="Page number for pagination", ge=1
    ),
    page_size: int = fastapi.Query(
        default=10, description="Number of teams per page", ge=1, le=100
    ),
    sort_by: Optional[str] = fastapi.Query(
        default=None,
        description="Column to sort by (e.g. 'team_id', 'team_alias', 'created_at')",
    ),
    sort_order: str = fastapi.Query(
        default="asc", description="Sort order ('asc' or 'desc')"
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a paginated list of teams with filtering and sorting options.

    Parameters:
        user_id: Optional[str]
            Only return teams which this user belongs to
        organization_id: Optional[str]
            Only return teams which belong to this organization
        team_id: Optional[str]
            Filter teams by exact team_id match
        team_alias: Optional[str]
            Filter teams by partial team_alias match
        page: int
            The page number to return
        page_size: int
            The number of items per page
        sort_by: Optional[str]
            Column to sort by (e.g. 'team_id', 'team_alias', 'created_at')
        sort_order: str
            Sort order ('asc' or 'desc')
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": f"No db connected. prisma client={prisma_client}"},
        )

    if not allowed_route_check_inside_route(
        user_api_key_dict=user_api_key_dict, requested_user_id=user_id
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Only admin users can query all teams/other teams. Your user role={}".format(
                    user_api_key_dict.user_role
                )
            },
        )

    if user_id is None and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        user_id = user_api_key_dict.user_id

    # Calculate skip and take for pagination
    skip = (page - 1) * page_size

    # Build where conditions based on provided parameters
    where_conditions: Dict[str, Any] = {}

    if team_id:
        where_conditions["team_id"] = team_id

    if team_alias:
        where_conditions["team_alias"] = {
            "contains": team_alias,
            "mode": "insensitive",  # Case-insensitive search
        }

    if organization_id:
        where_conditions["organization_id"] = organization_id

    if user_id:
        try:
            user_object = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user_id}
            )
        except Exception:
            raise HTTPException(
                status_code=404,
                detail={"error": f"User not found, passed user_id={user_id}"},
            )
        if user_object is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"User not found, passed user_id={user_id}"},
            )
        user_object_correct_type = LiteLLM_UserTable(**user_object.model_dump())
        # Find teams where this user is a member by checking members_with_roles array
        if team_id is None:
            where_conditions["team_id"] = {"in": user_object_correct_type.teams}
        elif team_id in user_object_correct_type.teams:
            where_conditions["team_id"] = team_id
        else:
            raise HTTPException(
                status_code=404,
                detail={"error": f"User is not a member of team_id={team_id}"},
            )

    # Build order_by conditions
    valid_sort_columns = ["team_id", "team_alias", "created_at"]
    order_by = None
    if sort_by and sort_by in valid_sort_columns:
        if sort_order.lower() not in ["asc", "desc"]:
            sort_order = "asc"
        order_by = {sort_by: sort_order.lower()}

    # Get teams with pagination
    teams = await prisma_client.db.litellm_teamtable.find_many(
        where=where_conditions,
        skip=skip,
        take=page_size,
        order=order_by if order_by else {"created_at": "desc"},  # Default sort
    )
    # Get total count for pagination
    total_count = await prisma_client.db.litellm_teamtable.count(where=where_conditions)

    # Calculate total pages
    total_pages = -(-total_count // page_size)  # Ceiling division

    return {
        "teams": [team.model_dump() for team in teams] if teams else [],
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@router.get(
    "/team/list", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def list_team(
    http_request: Request,
    user_id: Optional[str] = fastapi.Query(
        default=None, description="Only return teams which this 'user_id' belongs to"
    ),
    organization_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ```
    curl --location --request GET 'http://0.0.0.0:4000/team/list' \
        --header 'Authorization: Bearer sk-1234'
    ```

    Parameters:
    - user_id: str - Optional. If passed will only return teams that the user_id is a member of.
    - organization_id: str - Optional. If passed will only return teams that belong to the organization_id. Pass 'default_organization' to get all teams without organization_id.
    """
    from litellm.proxy.proxy_server import prisma_client

    if not allowed_route_check_inside_route(
        user_api_key_dict=user_api_key_dict, requested_user_id=user_id
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Only admin users can query all teams/other teams. Your user role={}".format(
                    user_api_key_dict.user_role
                )
            },
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    response = await prisma_client.db.litellm_teamtable.find_many(
        include={
            "litellm_model_table": True,
        }
    )

    filtered_response = []
    if user_id:
        # Get user object to access their teams array
        for team in response:
            if team.members_with_roles:
                for member in team.members_with_roles:
                    if (
                        "user_id" in member
                        and member["user_id"] is not None
                        and member["user_id"] == user_id
                    ):
                        filtered_response.append(team)
    else:
        filtered_response = response

    _team_ids = [team.team_id for team in filtered_response]
    returned_tm = await get_all_team_memberships(
        prisma_client, _team_ids, user_id=user_id
    )

    returned_responses: List[TeamListResponseObject] = []
    for team in filtered_response:
        _team_memberships: List[LiteLLM_TeamMembership] = []
        for tm in returned_tm:
            if tm.team_id == team.team_id:
                _team_memberships.append(tm)

        # add all keys that belong to the team
        keys = await prisma_client.db.litellm_verificationtoken.find_many(
            where={"team_id": team.team_id}
        )

        try:
            returned_responses.append(
                TeamListResponseObject(
                    **team.model_dump(),
                    team_memberships=_team_memberships,
                    keys=keys,
                )
            )
        except Exception as e:
            team_exception = """Invalid team object for team_id: {}. team_object={}.
            Error: {}
            """.format(
                team.team_id, team.model_dump(), str(e)
            )
            verbose_proxy_logger.exception(team_exception)
            continue
    # Sort the responses by team_alias
    returned_responses.sort(key=lambda x: (getattr(x, "team_alias", "") or ""))

    if organization_id is not None:
        if organization_id == SpecialManagementEndpointEnums.DEFAULT_ORGANIZATION.value:
            returned_responses = [
                team for team in returned_responses if team.organization_id is None
            ]
        else:
            returned_responses = [
                team
                for team in returned_responses
                if team.organization_id == organization_id
            ]

    return returned_responses


async def get_paginated_teams(
    prisma_client: PrismaClient,
    page_size: int = 10,
    page: int = 1,
) -> Tuple[List[LiteLLM_TeamTable], int]:
    """
    Get paginated list of teams from team table

    Parameters:
        prisma_client: PrismaClient - The database client
        page_size: int - Number of teams per page
        page: int - Page number (1-based)

    Returns:
        Tuple[List[LiteLLM_TeamTable], int] - (list of teams, total count)
    """
    try:
        # Calculate skip for pagination
        skip = (page - 1) * page_size
        # Get total count
        total_count = await prisma_client.db.litellm_teamtable.count()

        # Get paginated teams
        teams = await prisma_client.db.litellm_teamtable.find_many(
            skip=skip, take=page_size, order={"team_alias": "asc"}  # Sort by team_alias
        )
        return teams, total_count
    except Exception as e:
        verbose_proxy_logger.exception(
            f"[Non-Blocking] Error getting paginated teams: {e}"
        )
        return [], 0


@router.get(
    "/team/filter/ui",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
    responses={
        200: {"model": List[LiteLLM_TeamTable]},
    },
)
async def ui_view_teams(
    team_id: Optional[str] = fastapi.Query(
        default=None, description="Team ID in the request parameters"
    ),
    team_alias: Optional[str] = fastapi.Query(
        default=None, description="Team alias in the request parameters"
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
    [PROXY-ADMIN ONLY] Filter teams based on partial match of team_id or team_alias with pagination.

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

        if team_id:
            where_conditions["team_id"] = {
                "contains": team_id,
                "mode": "insensitive",  # Case-insensitive search
            }

        if team_alias:
            where_conditions["team_alias"] = {
                "contains": team_alias,
                "mode": "insensitive",  # Case-insensitive search
            }

        # Query users with pagination and filters
        teams = await prisma_client.db.litellm_teamtable.find_many(
            where=where_conditions,
            skip=skip,
            take=page_size,
            order={"created_at": "desc"},
        )

        if not teams:
            return []

        return teams

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching teams: {str(e)}")


def add_new_models_to_team(
    team_obj: LiteLLM_TeamTable, new_models: List[str]
) -> List[str]:
    """
    Add new models to a team's allowed model list.
    """
    current_models = team_obj.models
    if (
        current_models is not None and len(current_models) == 0
    ):  # implies all model access
        current_models = [SpecialModelNames.all_proxy_models.value]
    else:
        current_models = team_obj.models
    updated_models = list(set(current_models + new_models))
    return updated_models


@router.post(
    "/team/model/add",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def team_model_add(
    data: TeamModelAddRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Add models to a team's allowed model list. Only proxy admin or team admin can add models.

    Parameters:
    - team_id: str - Required. The team to add models to
    - models: List[str] - Required. List of models to add to the team

    Example Request:
    ```
    curl --location 'http://0.0.0.0:4000/team/model/add' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "team_id": "team-1234",
        "models": ["gpt-4", "claude-2"]
    }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    # Get existing team
    team_row = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": data.team_id}
    )

    if team_row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Team not found, passed team_id={data.team_id}"},
        )

    team_obj = LiteLLM_TeamTable(**team_row.model_dump())

    # Authorization check - only proxy admin or team admin can add models
    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(
            user_api_key_dict=user_api_key_dict, team_obj=team_obj
        )
    ):
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admin or team admin can modify team models"},
        )

    updated_models = add_new_models_to_team(team_obj=team_obj, new_models=data.models)
    # Update team
    updated_team = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id}, data={"models": updated_models}
    )

    return updated_team


@router.post(
    "/team/model/delete",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def team_model_delete(
    data: TeamModelDeleteRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Remove models from a team's allowed model list. Only proxy admin or team admin can remove models.

    Parameters:
    - team_id: str - Required. The team to remove models from
    - models: List[str] - Required. List of models to remove from the team

    Example Request:
    ```
    curl --location 'http://0.0.0.0:4000/team/model/delete' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "team_id": "team-1234",
        "models": ["gpt-4"]
    }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    # Get existing team
    team_row = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": data.team_id}
    )

    if team_row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Team not found, passed team_id={data.team_id}"},
        )

    team_obj = LiteLLM_TeamTable(**team_row.model_dump())

    # Authorization check - only proxy admin or team admin can remove models
    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(
            user_api_key_dict=user_api_key_dict, team_obj=team_obj
        )
    ):
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admin or team admin can modify team models"},
        )

    # Get current models list
    current_models = team_obj.models or []

    # Remove specified models
    updated_models = [m for m in current_models if m not in data.models]

    # Update team
    updated_team = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id}, data={"models": updated_models}
    )

    return updated_team


@router.get(
    "/team/permissions_list",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def team_member_permissions(
    team_id: str = fastapi.Query(
        default=None, description="Team ID in the request parameters"
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> GetTeamMemberPermissionsResponse:
    """
    Get the team member permissions for a team
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    ## CHECK IF USER IS PROXY ADMIN OR TEAM ADMIN
    existing_team_row = await get_team_object(
        team_id=team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=proxy_logging_obj,
        check_cache_only=False,
        check_db_only=True,
    )
    if existing_team_row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Team not found for team_id={team_id}"},
        )

    complete_team_data = LiteLLM_TeamTable(**existing_team_row.model_dump())

    if (
        hasattr(user_api_key_dict, "user_role")
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(
            user_api_key_dict=user_api_key_dict, team_obj=complete_team_data
        )
        and not _is_available_team(
            team_id=complete_team_data.team_id,
            user_api_key_dict=user_api_key_dict,
        )
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Call not allowed. User not proxy admin OR team admin. route={}, team_id={}".format(
                    "/team/member_add",
                    complete_team_data.team_id,
                )
            },
        )

    if existing_team_row.team_member_permissions is None:
        existing_team_row.team_member_permissions = (
            TeamMemberPermissionChecks.default_team_member_permissions()
        )

    return GetTeamMemberPermissionsResponse(
        team_id=team_id,
        team_member_permissions=existing_team_row.team_member_permissions,
        all_available_permissions=TeamMemberPermissionChecks.get_all_available_team_member_permissions(),
    )


@router.post(
    "/team/permissions_update",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_team_member_permissions(
    data: UpdateTeamMemberPermissionsRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> LiteLLM_TeamTable:
    """
    Update the team member permissions for a team
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    ## CHECK IF USER IS PROXY ADMIN OR TEAM ADMIN
    existing_team_row = await get_team_object(
        team_id=data.team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=proxy_logging_obj,
        check_cache_only=False,
        check_db_only=True,
    )
    if existing_team_row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Team not found for team_id={data.team_id}"},
        )

    complete_team_data = LiteLLM_TeamTable(**existing_team_row.model_dump())

    if (
        hasattr(user_api_key_dict, "user_role")
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(
            user_api_key_dict=user_api_key_dict, team_obj=complete_team_data
        )
        and not _is_available_team(
            team_id=complete_team_data.team_id,
            user_api_key_dict=user_api_key_dict,
        )
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Call not allowed. User not proxy admin OR team admin. route={}, team_id={}".format(
                    "/team/member_add",
                    complete_team_data.team_id,
                )
            },
        )
    # Update the team member permissions
    updated_team = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id},
        data={"team_member_permissions": data.team_member_permissions},
    )

    return updated_team


@router.get(
    "/team/daily/activity",
    response_model=SpendAnalyticsPaginatedResponse,
    tags=["team management"],
)
async def get_team_daily_activity(
    team_ids: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    exclude_team_ids: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get daily activity for specific teams or all teams.

    Args:
        team_ids (Optional[str]): Comma-separated list of team IDs to filter by. If not provided, returns data for all teams.
        start_date (Optional[str]): Start date for the activity period (YYYY-MM-DD).
        end_date (Optional[str]): End date for the activity period (YYYY-MM-DD).
        model (Optional[str]): Filter by model name.
        api_key (Optional[str]): Filter by API key.
        page (int): Page number for pagination.
        page_size (int): Number of items per page.
        exclude_team_ids (Optional[str]): Comma-separated list of team IDs to exclude.
    Returns:
        SpendAnalyticsPaginatedResponse: Paginated response containing daily activity data.
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Convert comma-separated tags string to list if provided
    team_ids_list = team_ids.split(",") if team_ids else None
    exclude_team_ids_list: Optional[List[str]] = None

    if exclude_team_ids:
        exclude_team_ids_list = (
            exclude_team_ids.split(",") if exclude_team_ids else None
        )

    if not _user_has_admin_view(user_api_key_dict):
        user_info = await get_user_object(
            user_id=user_api_key_dict.user_id,
            prisma_client=prisma_client,
            user_id_upsert=False,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
            check_db_only=True,
        )
        if user_info is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "User= {} not found".format(user_api_key_dict.user_id)
                },
            )

        if team_ids_list is None:
            team_ids_list = user_info.teams
        else:
            # check if all team_ids are in user_info.teams
            for team_id in team_ids_list:
                if team_id not in user_info.teams:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "error": "User does not belong to Team= {}. Call `/user/info` to see user's teams".format(
                                team_id
                            )
                        },
                    )

    ## Fetch team aliases
    where_condition = {}
    if team_ids_list:
        where_condition["team_id"] = {"in": list(team_ids_list)}
    team_aliases = await prisma_client.db.litellm_teamtable.find_many(
        where=where_condition
    )
    team_alias_metadata = {
        t.team_id: {"team_alias": t.team_alias} for t in team_aliases
    }

    return await get_daily_activity(
        prisma_client=prisma_client,
        table_name="litellm_dailyteamspend",
        entity_id_field="team_id",
        entity_id=team_ids_list,
        entity_metadata_field=team_alias_metadata,
        exclude_entity_ids=exclude_team_ids_list,
        start_date=start_date,
        end_date=end_date,
        model=model,
        api_key=api_key,
        page=page,
        page_size=page_size,
    )
