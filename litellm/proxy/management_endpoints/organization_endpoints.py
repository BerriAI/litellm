"""
Endpoints for /organization operations

/organization/new
/organization/update
/organization/delete
/organization/member_add
/organization/info
/organization/list
"""

#### ORGANIZATION MANAGEMENT ####

import uuid
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.auth_checks import can_user_call_model
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.budget_management_endpoints import (
    new_budget,
    update_budget,
)
from litellm.proxy.management_helpers.object_permission_utils import (
    handle_update_object_permission_common,
)
from litellm.proxy.management_helpers.utils import (
    get_new_internal_user_defaults,
    management_endpoint_wrapper,
)
from litellm.proxy.utils import PrismaClient

router = APIRouter()


@router.post(
    "/organization/new",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=NewOrganizationResponse,
)
async def new_organization(
    data: NewOrganizationRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Allow orgs to own teams

    Set org level budgets + model access.

    Only admins can create orgs.

    # Parameters

    - organization_alias: *str* - The name of the organization.
    - models: *List* - The models the organization has access to.
    - budget_id: *Optional[str]* - The id for a budget (tpm/rpm/max budget) for the organization.
    ### IF NO BUDGET ID - CREATE ONE WITH THESE PARAMS ###
    - max_budget: *Optional[float]* - Max budget for org
    - tpm_limit: *Optional[int]* - Max tpm limit for org
    - rpm_limit: *Optional[int]* - Max rpm limit for org
    - max_parallel_requests: *Optional[int]* - [Not Implemented Yet] Max parallel requests for org
    - soft_budget: *Optional[float]* - [Not Implemented Yet] Get a slack alert when this soft budget is reached. Don't block requests.
    - model_max_budget: *Optional[dict]* - Max budget for a specific model
    - budget_duration: *Optional[str]* - Frequency of reseting org budget
    - metadata: *Optional[dict]* - Metadata for organization, store information for organization. Example metadata - {"extra_info": "some info"}
    - blocked: *bool* - Flag indicating if the org is blocked or not - will stop all calls from keys with this org_id.
    - tags: *Optional[List[str]]* - Tags for [tracking spend](https://litellm.vercel.app/docs/proxy/enterprise#tracking-spend-for-custom-tags) and/or doing [tag-based routing](https://litellm.vercel.app/docs/proxy/tag_routing).
    - organization_id: *Optional[str]* - The organization id of the team. Default is None. Create via `/organization/new`.
    - model_aliases: Optional[dict] - Model aliases for the team. [Docs](https://docs.litellm.ai/docs/proxy/team_based_routing#create-team-with-model-alias)
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - organization-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"]}. IF null or {} then no object permission.
    Case 1: Create new org **without** a budget_id

    ```bash
    curl --location 'http://0.0.0.0:4000/organization/new' \

    --header 'Authorization: Bearer sk-1234' \

    --header 'Content-Type: application/json' \

    --data '{
        "organization_alias": "my-secret-org",
        "models": ["model1", "model2"],
        "max_budget": 100
    }'


    ```

    Case 2: Create new org **with** a budget_id

    ```bash
    curl --location 'http://0.0.0.0:4000/organization/new' \

    --header 'Authorization: Bearer sk-1234' \

    --header 'Content-Type: application/json' \

    --data '{
        "organization_alias": "my-secret-org",
        "models": ["model1", "model2"],
        "budget_id": "428eeaa8-f3ac-4e85-a8fb-7dc8d7aa8689"
    }'
    ```
    """

    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        llm_router,
        prisma_client,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if (
        user_api_key_dict.user_role is None
        or user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": f"Only admins can create orgs. Your role is = {user_api_key_dict.user_role}"
            },
        )

    if llm_router is None:
        raise HTTPException(
            status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
        )

    user_object_correct_type: Optional[LiteLLM_UserTable] = None

    if user_api_key_dict.user_id is not None:
        try:
            user_object = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user_api_key_dict.user_id}
            )
            user_object_correct_type = LiteLLM_UserTable(**user_object.model_dump())
        except Exception:
            pass

    if data.budget_id is None:
        """
        Every organization needs a budget attached.

        If none provided, create one based on provided values
        """
        budget_params = LiteLLM_BudgetTable.model_fields.keys()

        # Only include Budget Params when creating an entry in litellm_budgettable
        _json_data = data.json(exclude_none=True)
        _budget_data = {k: v for k, v in _json_data.items() if k in budget_params}
        budget_row = LiteLLM_BudgetTable(**_budget_data)

        new_budget = prisma_client.jsonify_object(budget_row.json(exclude_none=True))

        _budget = await prisma_client.db.litellm_budgettable.create(
            data={
                **new_budget,  # type: ignore
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )  # type: ignore

        data.budget_id = _budget.budget_id

    ## Handle Object Permission - MCP, Vector Stores etc.
    object_permission_id = await _set_object_permission(
        data=data,
        prisma_client=prisma_client,
    )

    """
    Ensure only models that user has access to, are given to org
    """
    if len(user_api_key_dict.models) == 0:  # user has access to all models
        pass
    else:
        if len(data.models) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "User not allowed to give access to all models. Select models you want org to have access to."
                },
            )

        for m in data.models:
            await can_user_call_model(
                m, llm_router=llm_router, user_object=user_object_correct_type
            )

    organization_row = LiteLLM_OrganizationTable(
        **data.json(exclude_none=True),
        object_permission_id=object_permission_id,
        created_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
        updated_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
    )
    new_organization_row = prisma_client.jsonify_object(
        organization_row.json(exclude_none=True)
    )
    verbose_proxy_logger.info(
        f"new_organization_row: {json.dumps(new_organization_row, indent=2)}"
    )
    response = await prisma_client.db.litellm_organizationtable.create(
        data={
            **new_organization_row,  # type: ignore
        }
    )

    return response


async def _set_object_permission(
    data: NewOrganizationRequest,
    prisma_client: Optional[PrismaClient],
) -> Optional[str]:
    """
    Creates the LiteLLM_ObjectPermissionTable record for the organization.
    - Handles permissions for vector stores and mcp servers.

    Returns the object_permission_id if created, otherwise None.
    """
    if prisma_client is None:
        return None

    if data.object_permission is not None:
        created_object_permission = (
            await prisma_client.db.litellm_objectpermissiontable.create(
                data=data.object_permission.model_dump(exclude_none=True),
            )
        )
        del data.object_permission
        return created_object_permission.object_permission_id
    return None


@router.patch(
    "/organization/update",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_OrganizationTableWithMembers,
)
async def update_organization(
    data: LiteLLM_OrganizationTableUpdate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an organization
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if user_api_key_dict.user_id is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Cannot associate a user_id to this action. Check `/key/info` to validate if 'user_id' is set."
            },
        )

    if data.updated_by is None:
        data.updated_by = user_api_key_dict.user_id

    updated_organization_row = prisma_client.jsonify_object(
        data.model_dump(exclude_none=True)
    )
    existing_organization_row = (
        await prisma_client.db.litellm_organizationtable.find_unique(
            where={"organization_id": data.organization_id},
        )
    )

    if existing_organization_row is None:
        raise ValueError(
            f"Organization not found for organization_id={data.organization_id}"
        )

    if data.object_permission is not None:
        updated_organization_row = await handle_update_object_permission(
            data_json=updated_organization_row,
            existing_organization_row=existing_organization_row,
        )

    response = await prisma_client.db.litellm_organizationtable.update(
        where={"organization_id": data.organization_id},
        data=updated_organization_row,
        include={"members": True, "teams": True, "litellm_budget_table": True},
    )

    return response


async def handle_update_object_permission(
    data_json: dict,
    existing_organization_row: LiteLLM_OrganizationTable,
) -> dict:
    """
    Handle the update of object permission for an organization.

    - Upserts the new object permission into the LiteLLM_ObjectPermissionTable
    - Adds object_permission_id to data_json (this gets added in the DB)
    - Pops the object_permission from data_json
    """
    from litellm.proxy.proxy_server import prisma_client

    # Use the common helper to handle the object permission update
    object_permission_id = await handle_update_object_permission_common(
        data_json=data_json,
        existing_object_permission_id=existing_organization_row.object_permission_id,
        prisma_client=prisma_client,
    )

    # Add the object_permission_id to data_json if one was created/updated
    if object_permission_id is not None:
        data_json["object_permission_id"] = object_permission_id

    return data_json


@router.delete(
    "/organization/delete",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[LiteLLM_OrganizationTableWithMembers],
)
async def delete_organization(
    data: DeleteOrganizationRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete an organization

    # Parameters:

    - organization_ids: List[str] - The organization ids to delete.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=401,
            detail={"error": "Only proxy admins can delete organizations"},
        )

    deleted_orgs = []
    for organization_id in data.organization_ids:
        # delete all teams in the organization
        await prisma_client.db.litellm_teamtable.delete_many(
            where={"organization_id": organization_id}
        )
        # delete all members in the organization
        await prisma_client.db.litellm_organizationmembership.delete_many(
            where={"organization_id": organization_id}
        )
        # delete all keys in the organization
        await prisma_client.db.litellm_verificationtoken.delete_many(
            where={"organization_id": organization_id}
        )
        # delete the organization
        deleted_org = await prisma_client.db.litellm_organizationtable.delete(
            where={"organization_id": organization_id},
            include={"members": True, "teams": True, "litellm_budget_table": True},
        )
        if deleted_org is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Organization={organization_id} not found"},
            )
        deleted_orgs.append(deleted_org)

    return deleted_orgs


@router.get(
    "/organization/list",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[LiteLLM_OrganizationTableWithMembers],
)
async def list_organization(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ```
    curl --location --request GET 'http://0.0.0.0:4000/organization/list' \
        --header 'Authorization: Bearer sk-1234'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # if proxy admin - get all orgs
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        response = await prisma_client.db.litellm_organizationtable.find_many(
            include={"litellm_budget_table": True, "members": True, "teams": True}
        )
    # if internal user - get orgs they are a member of
    else:
        org_memberships = (
            await prisma_client.db.litellm_organizationmembership.find_many(
                where={"user_id": user_api_key_dict.user_id}
            )
        )
        org_objects = await prisma_client.db.litellm_organizationtable.find_many(
            where={
                "organization_id": {
                    "in": [membership.organization_id for membership in org_memberships]
                }
            },
            include={"litellm_budget_table": True, "members": True, "teams": True},
        )

        response = org_objects

    return response


@router.get(
    "/organization/info",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_OrganizationTableWithMembers,
)
async def info_organization(organization_id: str):
    """
    Get the org specific information
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    response: Optional[
        LiteLLM_OrganizationTableWithMembers
    ] = await prisma_client.db.litellm_organizationtable.find_unique(
        where={"organization_id": organization_id},
        include={
            "litellm_budget_table": True,
            "members": True,
            "teams": True,
            "object_permission": True,
        },
    )

    if response is None:
        raise HTTPException(status_code=404, detail={"error": "Organization not found"})

    response_pydantic_obj = LiteLLM_OrganizationTableWithMembers(
        **response.model_dump()
    )

    return response_pydantic_obj


@router.post(
    "/organization/info",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def deprecated_info_organization(data: OrganizationRequest):
    """
    DEPRECATED: Use GET /organization/info instead
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if len(data.organizations) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Specify list of organization id's to query. Passed in={data.organizations}"
            },
        )
    response = await prisma_client.db.litellm_organizationtable.find_many(
        where={"organization_id": {"in": data.organizations}},
        include={"litellm_budget_table": True},
    )

    return response


@router.post(
    "/organization/member_add",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=OrganizationAddMemberResponse,
)
@management_endpoint_wrapper
async def organization_member_add(
    data: OrganizationMemberAddRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> OrganizationAddMemberResponse:
    """
    [BETA]

    Add new members (either via user_email or user_id) to an organization

    If user doesn't exist, new user row will also be added to User Table

    Only proxy_admin or org_admin of organization, allowed to access this endpoint.

    # Parameters:

    - organization_id: str (required)
    - member: Union[List[Member], Member] (required)
        - role: Literal[LitellmUserRoles] (required)
        - user_id: Optional[str]
        - user_email: Optional[str]

    Note: Either user_id or user_email must be provided for each member.

    Example:
    ```
    curl -X POST 'http://0.0.0.0:4000/organization/member_add' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -d '{
        "organization_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849",
        "member": {
            "role": "internal_user",
            "user_id": "krrish247652@berri.ai"
        },
        "max_budget_in_organization": 100.0
    }'
    ```

    The following is executed in this function:

    1. Check if organization exists
    2. Creates a new Internal User if the user_id or user_email is not found in LiteLLM_UserTable
    3. Add Internal User to the `LiteLLM_OrganizationMembership` table
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

        # Check if organization exists
        existing_organization_row = (
            await prisma_client.db.litellm_organizationtable.find_unique(
                where={"organization_id": data.organization_id}
            )
        )
        if existing_organization_row is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Organization not found for organization_id={getattr(data, 'organization_id', None)}"
                },
            )

        members: List[OrgMember]
        if isinstance(data.member, List):
            members = data.member
        else:
            members = [data.member]

        updated_users: List[LiteLLM_UserTable] = []
        updated_organization_memberships: List[LiteLLM_OrganizationMembershipTable] = []

        for member in members:
            (
                updated_user,
                updated_organization_membership,
            ) = await add_member_to_organization(
                member=member,
                organization_id=data.organization_id,
                prisma_client=prisma_client,
            )

            updated_users.append(updated_user)
            updated_organization_memberships.append(updated_organization_membership)

        return OrganizationAddMemberResponse(
            organization_id=data.organization_id,
            updated_users=updated_users,
            updated_organization_memberships=updated_organization_memberships,
        )
    except Exception as e:
        verbose_proxy_logger.exception(f"Error adding member to organization: {e}")
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def find_member_if_email(
    user_email: str, prisma_client: PrismaClient
) -> LiteLLM_UserTable:
    """
    Find a member if the user_email is in LiteLLM_UserTable
    """

    try:
        existing_user_email_row: BaseModel = (
            await prisma_client.db.litellm_usertable.find_unique(
                where={"user_email": user_email}
            )
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unique user not found for user_email={user_email}. Potential duplicate OR non-existent user_email in LiteLLM_UserTable. Use 'user_id' instead."
            },
        )
    existing_user_email_row_pydantic = LiteLLM_UserTable(
        **existing_user_email_row.model_dump()
    )
    return existing_user_email_row_pydantic


@router.patch(
    "/organization/member_update",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_OrganizationMembershipTable,
)
@management_endpoint_wrapper
async def organization_member_update(
    data: OrganizationMemberUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update a member's role in an organization
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        # Check if organization exists
        existing_organization_row = (
            await prisma_client.db.litellm_organizationtable.find_unique(
                where={"organization_id": data.organization_id}
            )
        )
        if existing_organization_row is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Organization not found for organization_id={getattr(data, 'organization_id', None)}"
                },
            )

        # Check if member exists in organization
        if data.user_email is not None and data.user_id is None:
            existing_user_email_row = await find_member_if_email(
                data.user_email, prisma_client
            )
            data.user_id = existing_user_email_row.user_id

        try:
            existing_organization_membership = (
                await prisma_client.db.litellm_organizationmembership.find_unique(
                    where={
                        "user_id_organization_id": {
                            "user_id": data.user_id,
                            "organization_id": data.organization_id,
                        }
                    }
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Error finding organization membership for user_id={data.user_id} in organization={data.organization_id}: {e}"
                },
            )
        if existing_organization_membership is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Member not found in organization for user_id={data.user_id}"
                },
            )

        # Update member role
        if data.role is not None:
            await prisma_client.db.litellm_organizationmembership.update(
                where={
                    "user_id_organization_id": {
                        "user_id": data.user_id,
                        "organization_id": data.organization_id,
                    }
                },
                data={"user_role": data.role},
            )
        if data.max_budget_in_organization is not None:
            # if budget_id is None, create a new budget
            budget_id = existing_organization_membership.budget_id or str(uuid.uuid4())
            if existing_organization_membership.budget_id is None:
                new_budget_obj = BudgetNewRequest(
                    budget_id=budget_id, max_budget=data.max_budget_in_organization
                )
                await new_budget(
                    budget_obj=new_budget_obj, user_api_key_dict=user_api_key_dict
                )
            else:
                # update budget table with new max_budget
                await update_budget(
                    budget_obj=BudgetNewRequest(
                        budget_id=budget_id, max_budget=data.max_budget_in_organization
                    ),
                    user_api_key_dict=user_api_key_dict,
                )

            # update organization membership with new budget_id
            await prisma_client.db.litellm_organizationmembership.update(
                where={
                    "user_id_organization_id": {
                        "user_id": data.user_id,
                        "organization_id": data.organization_id,
                    }
                },
                data={"budget_id": budget_id},
            )
        final_organization_membership: Optional[
            BaseModel
        ] = await prisma_client.db.litellm_organizationmembership.find_unique(
            where={
                "user_id_organization_id": {
                    "user_id": data.user_id,
                    "organization_id": data.organization_id,
                }
            },
            include={"litellm_budget_table": True},
        )

        if final_organization_membership is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Member not found in organization={data.organization_id} for user_id={data.user_id}"
                },
            )

        final_organization_membership_pydantic = LiteLLM_OrganizationMembershipTable(
            **final_organization_membership.model_dump(exclude_none=True)
        )
        return final_organization_membership_pydantic
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating member in organization: {e}")
        raise e


@router.delete(
    "/organization/member_delete",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def organization_member_delete(
    data: OrganizationMemberDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a member from an organization
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        if data.user_email is not None and data.user_id is None:
            existing_user_email_row = await find_member_if_email(
                data.user_email, prisma_client
            )
            data.user_id = existing_user_email_row.user_id

        member_to_delete = await prisma_client.db.litellm_organizationmembership.delete(
            where={
                "user_id_organization_id": {
                    "user_id": data.user_id,
                    "organization_id": data.organization_id,
                }
            }
        )
        return member_to_delete

    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting member from organization: {e}")
        raise e


async def add_member_to_organization(
    member: OrgMember,
    organization_id: str,
    prisma_client: PrismaClient,
) -> Tuple[LiteLLM_UserTable, LiteLLM_OrganizationMembershipTable]:
    """
    Add a member to an organization

    - Checks if member.user_id or member.user_email is in LiteLLM_UserTable
    - If not found, create a new user in LiteLLM_UserTable
    - Add user to organization in LiteLLM_OrganizationMembership
    """

    try:
        user_object: Optional[LiteLLM_UserTable] = None
        existing_user_id_row = None
        existing_user_email_row = None
        ## Check if user exists in LiteLLM_UserTable - user exists - either the user_id or user_email is in LiteLLM_UserTable
        if member.user_id is not None:
            existing_user_id_row = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": member.user_id}
            )

        if existing_user_id_row is None and member.user_email is not None:
            try:
                existing_user_email_row = (
                    await prisma_client.db.litellm_usertable.find_unique(
                        where={"user_email": member.user_email}
                    )
                )
            except Exception as e:
                raise ValueError(
                    f"Potential NON-Existent or Duplicate user email in DB: Error finding a unique instance of user_email={member.user_email} in LiteLLM_UserTable.: {e}"
                )

        ## If user does not exist, create a new user
        if existing_user_id_row is None and existing_user_email_row is None:
            # Create a new user - since user does not exist
            user_id: str = member.user_id or str(uuid.uuid4())
            new_user_defaults = get_new_internal_user_defaults(
                user_id=user_id,
                user_email=member.user_email,
            )

            _returned_user = await prisma_client.insert_data(data=new_user_defaults, table_name="user")  # type: ignore
            if _returned_user is not None:
                user_object = LiteLLM_UserTable(**_returned_user.model_dump())
        elif existing_user_email_row is not None and len(existing_user_email_row) > 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Multiple users with this email found in db. Please use 'user_id' instead."
                },
            )
        elif existing_user_email_row is not None:
            user_object = LiteLLM_UserTable(**existing_user_email_row.model_dump())
        elif existing_user_id_row is not None:
            user_object = LiteLLM_UserTable(**existing_user_id_row.model_dump())
        else:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"User not found for user_id={member.user_id} and user_email={member.user_email}"
                },
            )

        if user_object is None:
            raise ValueError(
                f"User does not exist in LiteLLM_UserTable. user_id={member.user_id} and user_email={member.user_email}"
            )

        # Add user to organization
        _organization_membership = (
            await prisma_client.db.litellm_organizationmembership.create(
                data={
                    "organization_id": organization_id,
                    "user_id": user_object.user_id,
                    "user_role": member.role,
                }
            )
        )
        organization_membership = LiteLLM_OrganizationMembershipTable(
            **_organization_membership.model_dump()
        )
        return user_object, organization_membership

    except Exception as e:
        raise ValueError(
            f"Error adding member={member} to organization={organization_id}: {e}"
        )
