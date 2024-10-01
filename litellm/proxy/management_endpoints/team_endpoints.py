import asyncio
import copy
import json
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

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
    TeamAddMemberResponse,
    TeamBase,
    TeamInfoResponseObject,
    TeamMemberAddRequest,
    TeamMemberDeleteRequest,
    TeamMemberUpdateRequest,
    TeamMemberUpdateResponse,
    UpdateTeamRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import get_team_object
from litellm.proxy.auth.user_api_key_auth import _is_user_proxy_admin, user_api_key_auth
from litellm.proxy.management_helpers.utils import (
    add_new_member,
    management_endpoint_wrapper,
)

router = APIRouter()


def _is_user_team_admin(
    user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable
) -> bool:
    for member in team_obj.members_with_roles:
        if member.user_id is not None and member.user_id == user_api_key_dict.user_id:
            return True

    return False


#### TEAM MANAGEMENT ####
@router.post(
    "/team/new",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_TeamTable,
)
@management_endpoint_wrapper
async def new_team(
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
    - metadata: Optional[dict] - Metadata for team, store information for team. Example metadata = {"extra_info": "some info"}
    - tpm_limit: Optional[int] - The TPM (Tokens Per Minute) limit for this team - all keys with this team_id will have at max this TPM limit
    - rpm_limit: Optional[int] - The RPM (Requests Per Minute) limit for this team - all keys associated with this team_id will have at max this RPM limit
    - max_budget: Optional[float] - The maximum budget allocated to the team - all keys for this team_id will have at max this max_budget
    - budget_duration: Optional[str] - The duration of the budget for the team. Doc [here](https://docs.litellm.ai/docs/proxy/team_budgets)
    - models: Optional[list] - A list of models associated with the team - all keys for this team_id will have at most, these models. If empty, assumes all models are allowed.
    - blocked: bool - Flag indicating if the team is blocked or not - will stop all calls from keys with this team_id.

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
    from litellm.proxy.proxy_server import (
        _duration_in_seconds,
        create_audit_log_for_update,
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

    # Set tags on the new team
    if data.tags is not None:
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            raise ValueError(
                f"Only premium users can add tags to teams. {CommonProxyErrors.not_premium_user.value}"
            )
        if complete_team_data.metadata is None:
            complete_team_data.metadata = {"tags": data.tags}
        else:
            complete_team_data.metadata["tags"] = data.tags

    # If budget_duration is set, set `budget_reset_at`
    if complete_team_data.budget_duration is not None:
        duration_s = _duration_in_seconds(duration=complete_team_data.budget_duration)
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)
        complete_team_data.budget_reset_at = reset_at

    team_row: LiteLLM_TeamTable = await prisma_client.insert_data(  # type: ignore
        data=complete_team_data.json(exclude_none=True), table_name="team"
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
    - metadata: Optional[dict] - Metadata for team, store information for team. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
    - tpm_limit: Optional[int] - The TPM (Tokens Per Minute) limit for this team - all keys with this team_id will have at max this TPM limit
    - rpm_limit: Optional[int] - The RPM (Requests Per Minute) limit for this team - all keys associated with this team_id will have at max this RPM limit
    - max_budget: Optional[float] - The maximum budget allocated to the team - all keys for this team_id will have at max this max_budget
    - budget_duration: Optional[str] - The duration of the budget for the team. Doc [here](https://docs.litellm.ai/docs/proxy/team_budgets)
    - models: Optional[list] - A list of models associated with the team - all keys for this team_id will have at most, these models. If empty, assumes all models are allowed.
    - blocked: bool - Flag indicating if the team is blocked or not - will stop all calls from keys with this team_id.

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
        _duration_in_seconds,
        create_audit_log_for_update,
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

    updated_kv = data.json(exclude_none=True)

    # Check budget_duration and budget_reset_at
    if data.budget_duration is not None:
        duration_s = _duration_in_seconds(duration=data.budget_duration)
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

        # set the budget_reset_at in DB
        updated_kv["budget_reset_at"] = reset_at

    # check if user is trying to update tags for team
    if "tags" in updated_kv and updated_kv["tags"] is not None:
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            raise ValueError(
                f"Only premium users can add tags to teams. {CommonProxyErrors.not_premium_user.value}"
            )
        # remove tags from updated_kv
        _tags = updated_kv.pop("tags")
        if "metadata" in updated_kv and updated_kv["metadata"] is not None:
            updated_kv["metadata"]["tags"] = _tags
        else:
            updated_kv["metadata"] = {"tags": _tags}

    updated_kv = prisma_client.jsonify_object(data=updated_kv)
    team_row: Optional[
        LiteLLM_TeamTable
    ] = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id}, data=updated_kv  # type: ignore
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
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_id is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    if data.member is None:
        raise HTTPException(
            status_code=400, detail={"error": "No member/members passed in"}
        )

    existing_team_row = await get_team_object(
        team_id=data.team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=proxy_logging_obj,
        check_cache_only=False,
    )
    if existing_team_row is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Team not found for team_id={getattr(data, 'team_id', None)}"
            },
        )

    complete_team_data = LiteLLM_TeamTable(**existing_team_row.model_dump())

    ## CHECK IF USER IS PROXY ADMIN OR TEAM ADMIN

    if (
        hasattr(user_api_key_dict, "user_role")
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(
            user_api_key_dict=user_api_key_dict, team_obj=complete_team_data
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
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA]

    delete members (either via user_email or user_id) from a team

    If user doesn't exist, an exception will be raised
    ```
    curl -X POST 'http://0.0.0.0:8000/team/update' \

    -H 'Authorization: Bearer sk-1234' \

    -H 'Content-Type: application/json' \

    -D '{
        "team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849",
        "user_id": "krrish247652@berri.ai"
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        _duration_in_seconds,
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

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
    new_team_members: List[Member] = []
    for m in existing_team_row.members_with_roles:
        if (
            data.user_id is not None
            and m.user_id is not None
            and data.user_id == m.user_id
        ):
            continue
        elif (
            data.user_email is not None
            and m.user_email is not None
            and data.user_email == m.user_email
        ):
            continue
        new_team_members.append(m)
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

    Update team member budgets
    """
    from litellm.proxy.proxy_server import (
        _duration_in_seconds,
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

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
    else:
        await prisma_client.db.litellm_budgettable.update(
            where={"budget_id": identified_budget_id},
            data={"max_budget": data.max_budget_in_team},
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
        _duration_in_seconds,
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_ids is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    # check that all teams passed exist
    for team_id in data.team_ids:
        team_row = await prisma_client.get_data(  # type: ignore
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if team_row is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team not found, passed team_id={team_id}"},
            )

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
    ## DELETE TEAMS
    deleted_teams = await prisma_client.delete_data(
        team_id_list=data.team_ids, table_name="team"
    )
    return deleted_teams


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

    ```
    curl --location 'http://localhost:4000/team/info?team_id=your_team_id_here' \
    --header 'Authorization: Bearer your_api_key_here'
    ```
    """
    from litellm.proxy.proxy_server import (
        _duration_in_seconds,
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

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

        if (
            user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
            or user_api_key_dict.user_role
            == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value
        ):
            pass
        elif user_api_key_dict.team_id is None or (
            team_id != user_api_key_dict.team_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="key not allowed to access this team's info. Key team_id={}, Requested team_id={}".format(
                    user_api_key_dict.team_id, team_id
                ),
            )

        team_info: Optional[Union[LiteLLM_TeamTable, dict]] = (
            await prisma_client.get_data(
                team_id=team_id, table_name="team", query_type="find_unique"
            )
        )
        if team_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": f"Team not found, passed team id: {team_id}."},
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
        team_memberships = await prisma_client.db.litellm_teammembership.find_many(
            where={"team_id": team_id},
            include={"litellm_budget_table": True},
        )

        returned_tm: List[LiteLLM_TeamMembership] = []
        for tm in team_memberships:
            returned_tm.append(LiteLLM_TeamMembership(**tm.model_dump()))

        if isinstance(team_info, dict):
            _team_info = LiteLLM_TeamTable(**team_info)
        elif isinstance(team_info, BaseModel):
            _team_info = LiteLLM_TeamTable(**team_info.model_dump())
        else:
            _team_info = LiteLLM_TeamTable()

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
    """
    from litellm.proxy.proxy_server import (
        _duration_in_seconds,
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

    if prisma_client is None:
        raise Exception("No DB Connected.")

    record = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id}, data={"blocked": True}  # type: ignore
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
    """
    from litellm.proxy.proxy_server import (
        _duration_in_seconds,
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

    if prisma_client is None:
        raise Exception("No DB Connected.")

    record = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id}, data={"blocked": False}  # type: ignore
    )

    return record


@router.get(
    "/team/list", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def list_team(
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Admin-only] List all available teams

    ```
    curl --location --request GET 'http://0.0.0.0:4000/team/list' \
        --header 'Authorization: Bearer sk-1234'
    ```
    """
    from litellm.proxy.proxy_server import (
        _duration_in_seconds,
        create_audit_log_for_update,
        litellm_proxy_admin_name,
        prisma_client,
    )

    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Admin-only endpoint. Your user role={}".format(
                    user_api_key_dict.user_role
                )
            },
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    response = await prisma_client.db.litellm_teamtable.find_many()

    return response
