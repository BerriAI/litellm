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
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Union, cast

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    BlockTeamRequest,
    CommonProxyErrors,
    DeleteTeamRequest,
    LiteLLM_AuditLogs,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    LiteLLM_ModelTable,
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
    TeamAddMemberResponse,
    TeamInfoResponseObject,
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
    get_team_object,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_team_admin,
    _set_object_metadata_field,
)
from litellm.proxy.management_endpoints.tag_management_endpoints import (
    get_daily_activity,
)
from litellm.proxy.management_helpers.team_member_permission_checks import (
    TeamMemberPermissionChecks,
)
from litellm.proxy.management_helpers.utils import (
    add_new_member,
    management_endpoint_wrapper,
)
from litellm.proxy.utils import (
    PrismaClient,
    _premium_user_check,
    handle_exception_on_proxy,
)
from litellm.router import Router
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.management_endpoints.team_endpoints import (
    GetTeamMemberPermissionsResponse,
    UpdateTeamMemberPermissionsRequest,
)

router = APIRouter()


def _is_available_team(team_id: str, user_api_key_dict: UserAPIKeyAuth) -> bool:
    if litellm.default_internal_user_params is None:
        return False
    if "available_teams" in litellm.default_internal_user_params:
        return team_id in litellm.default_internal_user_params["available_teams"]
    return False


async def get_all_team_memberships(
    prisma_client: PrismaClient, team_id: List[str], user_id: Optional[str] = None
) -> List[LiteLLM_TeamMembership]:
    """Get all team memberships for a given user"""
    ## GET ALL MEMBERSHIPS ##
    if not isinstance(user_id, str):
        user_id = str(user_id)

    team_memberships = await prisma_client.db.litellm_teammembership.find_many(
        where=(
            {"user_id": user_id, "team_id": {"in": team_id}}
            if user_id is not None
            else {"team_id": {"in": team_id}}
        ),
        include={"litellm_budget_table": True},
    )

    returned_tm: List[LiteLLM_TeamMembership] = []
    for tm in team_memberships:
        returned_tm.append(LiteLLM_TeamMembership(**tm.model_dump()))

    return returned_tm


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
    - tpm_limit: Optional[int] - The TPM (Tokens Per Minute) limit for this team - all keys with this team_id will have at max this TPM limit
    - rpm_limit: Optional[int] - The RPM (Requests Per Minute) limit for this team - all keys associated with this team_id will have at max this RPM limit
    - max_budget: Optional[float] - The maximum budget allocated to the team - all keys for this team_id will have at max this max_budget
    - budget_duration: Optional[str] - The duration of the budget for the team. Doc [here](https://docs.litellm.ai/docs/proxy/team_budgets)
    - models: Optional[list] - A list of models associated with the team - all keys for this team_id will have at most, these models. If empty, assumes all models are allowed.
    - blocked: bool - Flag indicating if the team is blocked or not - will stop all calls from keys with this team_id.
    - members: Optional[List] - Control team members via `/team/member/add` and `/team/member/delete`. 
    - tags: Optional[List[str]] - Tags for [tracking spend](https://litellm.vercel.app/docs/proxy/enterprise#tracking-spend-for-custom-tags) and/or doing [tag-based routing](https://litellm.vercel.app/docs/proxy/tag_routing).
    - organization_id: Optional[str] - The organization id of the team. Default is None. Create via `/organization/new`.
    - model_aliases: Optional[dict] - Model aliases for the team. [Docs](https://docs.litellm.ai/docs/proxy/team_based_routing#create-team-with-model-alias)
    - guardrails: Optional[List[str]] - Guardrails for the team. [Docs](https://docs.litellm.ai/docs/proxy/guardrails)
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
            create_audit_log_for_update,
            duration_in_seconds,
            litellm_proxy_admin_name,
            prisma_client,
        )

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

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

            if (
                data.max_budget is not None
                and user_api_key_dict.max_budget is not None
                and data.max_budget > user_api_key_dict.max_budget
            ):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"max budget higher than user max. User max budget={user_api_key_dict.max_budget}. User role={user_api_key_dict.user_role}"
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

        ## ADD TO TEAM TABLE
        complete_team_data = LiteLLM_TeamTable(
            **data.json(),
            model_id=_model_id,
        )

        # Set Management Endpoint Metadata Fields
        for field in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
            if getattr(data, field) is not None:
                _set_object_metadata_field(
                    object_data=complete_team_data,
                    field_name=field,
                    value=getattr(data, field),
                )

        # If budget_duration is set, set `budget_reset_at`
        if complete_team_data.budget_duration is not None:
            duration_s = duration_in_seconds(
                duration=complete_team_data.budget_duration
            )
            reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)
            complete_team_data.budget_reset_at = reset_at

        complete_team_data_dict = complete_team_data.model_dump(exclude_none=True)
        complete_team_data_dict = prisma_client.jsonify_team_object(
            db_data=complete_team_data_dict
        )
        team_row: LiteLLM_TeamTable = await prisma_client.db.litellm_teamtable.create(
            data=complete_team_data_dict,
            include={"litellm_model_table": True},  # type: ignore
        )

        ## ADD TEAM ID TO USER TABLE ##
        for user in complete_team_data.members_with_roles:
            ## add team id to user row ##
            await prisma_client.update_data(
                user_id=user.user_id,
                data={"user_id": user.user_id, "teams": [team_row.team_id]},
                update_key_values_custom_query={
                    "teams": {
                        "push ": [team_row.team_id],
                    }
                },
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
    - blocked: bool - Flag indicating if the team is blocked or not - will stop all calls from keys with this team_id.
    - tags: Optional[List[str]] - Tags for [tracking spend](https://litellm.vercel.app/docs/proxy/enterprise#tracking-spend-for-custom-tags) and/or doing [tag-based routing](https://litellm.vercel.app/docs/proxy/tag_routing).
    - organization_id: Optional[str] - The organization id of the team. Default is None. Create via `/organization/new`.
    - model_aliases: Optional[dict] - Model aliases for the team. [Docs](https://docs.litellm.ai/docs/proxy/team_based_routing#create-team-with-model-alias)
    - guardrails: Optional[List[str]] - Guardrails for the team. [Docs](https://docs.litellm.ai/docs/proxy/guardrails)
    Example - update team TPM Limit

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
        create_audit_log_for_update,
        duration_in_seconds,
        litellm_proxy_admin_name,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

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

    updated_kv = data.json(exclude_unset=True)

    # Check budget_duration and budget_reset_at
    if data.budget_duration is not None:
        duration_s = duration_in_seconds(duration=data.budget_duration)
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

        # set the budget_reset_at in DB
        updated_kv["budget_reset_at"] = reset_at

    # update team metadata fields
    _team_metadata_fields = LiteLLM_ManagementEndpoint_MetadataFields_Premium
    for field in _team_metadata_fields:
        if field in updated_kv and updated_kv[field] is not None:
            _update_team_metadata_field(
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
    team_row: Optional[
        LiteLLM_TeamTable
    ] = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id},
        data=updated_kv,
        include={"litellm_model_table": True},  # type: ignore
    )

    if team_row is None or team_row.team_id is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Team doesn't exist. Got={}".format(team_row)},
        )

    await _cache_team_object(
        team_id=team_row.team_id,
        team_table=LiteLLM_TeamTableCachedObj(**team_row.model_dump()),
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
    if litellm.store_audit_logs is True:
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
                    object_id=data.team_id,
                    action="updated",
                    updated_values=_after_value,
                    before_value=_before_value,
                )
            )
        )

    return {"team_id": team_row.team_id, "data": team_row}


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
    def _check_member_duplication(member: Member):
        if member.user_id in [m.user_id for m in existing_team_row.members_with_roles]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"User={member.user_id} already in team. Existing members={existing_team_row.members_with_roles}"
                },
            )

    if isinstance(data.member, Member):
        _check_member_duplication(data.member)
    elif isinstance(data.member, List):
        for m in data.member:
            _check_member_duplication(m)


@router.post(
    "/team/member_add",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=TeamAddMemberResponse,
)
@management_endpoint_wrapper
async def team_member_add(
    data: TeamMemberAddRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA]

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

    ## CHECK IF USER IS PROXY ADMIN OR TEAM ADMIN

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

    updated_users: List[LiteLLM_UserTable] = []
    updated_team_memberships: List[LiteLLM_TeamMembership] = []

    ## VALIDATE IF NEW MEMBER ##
    if isinstance(data.member, Member):
        try:
            updated_user, updated_tm = await add_new_member(
                new_member=data.member,
                max_budget_in_team=data.max_budget_in_team,
                prisma_client=prisma_client,
                user_api_key_dict=user_api_key_dict,
                litellm_proxy_admin_name=litellm_proxy_admin_name,
                team_id=data.team_id,
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
        tasks: List = []
        for m in data.member:
            try:
                updated_user, updated_tm = await add_new_member(
                    new_member=m,
                    max_budget_in_team=data.max_budget_in_team,
                    prisma_client=prisma_client,
                    user_api_key_dict=user_api_key_dict,
                    litellm_proxy_admin_name=litellm_proxy_admin_name,
                    team_id=data.team_id,
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

        await asyncio.gather(*tasks)

    ## ADD TO TEAM ##
    if isinstance(data.member, Member):
        # add to team db
        new_member = data.member

        # get user id
        if new_member.user_id is None and new_member.user_email is not None:
            for user in updated_users:
                if (
                    user.user_email is not None
                    and user.user_email == new_member.user_email
                ):
                    new_member.user_id = user.user_id

        complete_team_data.members_with_roles.append(new_member)

    elif isinstance(data.member, List):
        # add to team db
        new_members = data.member

        for nm in new_members:
            if nm.user_id is None and nm.user_email is not None:
                for user in updated_users:
                    if user.user_email is not None and user.user_email == nm.user_email:
                        nm.user_id = user.user_id

        complete_team_data.members_with_roles.extend(new_members)

    # ADD MEMBER TO TEAM
    _db_team_members = [m.model_dump() for m in complete_team_data.members_with_roles]
    updated_team = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id},
        data={"members_with_roles": json.dumps(_db_team_members)},  # type: ignore
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
    if data.max_budget_in_team is not None:
        if identified_budget_id is None:
            new_budget = await prisma_client.db.litellm_budgettable.create(
                data={
                    "max_budget": data.max_budget_in_team,
                    "created_by": user_api_key_dict.user_id or "",
                    "updated_by": user_api_key_dict.user_id or "",
                }
            )

            await prisma_client.db.litellm_teammembership.create(
                data={
                    "team_id": data.team_id,
                    "user_id": received_user_id,
                    "budget_id": new_budget.budget_id,
                },
            )
        elif identified_budget_id is not None:
            await prisma_client.db.litellm_budgettable.update(
                where={"budget_id": identified_budget_id},
                data={"max_budget": data.max_budget_in_team},
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
            team_row_base: Optional[
                BaseModel
            ] = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": team_id}
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
            team_info: Optional[
                BaseModel
            ] = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": team_id}
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
            _team_info = LiteLLM_TeamTable(**team_info)
        elif isinstance(team_info, BaseModel):
            _team_info = LiteLLM_TeamTable(**team_info.model_dump())
        else:
            _team_info = LiteLLM_TeamTable()

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


def _update_team_metadata_field(updated_kv: dict, field_name: str) -> None:
    """
    Helper function to update metadata fields that require premium user checks in the update endpoint

    Args:
        updated_kv: The key-value dict being used for the update
        field_name: Name of the metadata field being updated
    """
    if field_name in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
        _premium_user_check()

    if field_name in updated_kv and updated_kv[field_name] is not None:
        # remove field from updated_kv
        _value = updated_kv.pop(field_name)
        if "metadata" in updated_kv and updated_kv["metadata"] is not None:
            updated_kv["metadata"][field_name] = _value
        else:
            updated_kv["metadata"] = {field_name: _value}


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

    # Get current models list
    current_models = team_obj.models or []

    # Add new models (avoid duplicates)
    updated_models = list(set(current_models + data.models))

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
    dependencies=[Depends(user_api_key_auth)],
)
async def get_team_daily_activity(
    team_ids: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
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

    Returns:
        SpendAnalyticsPaginatedResponse: Paginated response containing daily activity data.
    """
    from litellm.proxy.proxy_server import prisma_client

    # Convert comma-separated tags string to list if provided
    team_ids_list = team_ids.split(",") if team_ids else None

    return await get_daily_activity(
        prisma_client=prisma_client,
        table_name="litellm_dailyteamspend",
        entity_id_field="team_id",
        entity_id=team_ids_list,
        start_date=start_date,
        end_date=end_date,
        model=model,
        api_key=api_key,
        page=page,
        page_size=page_size,
    )
