"""
Endpoints for /organization operations

/organization/new
/organization/update
/organization/delete
/organization/info
/organization/list
"""

#### ORGANIZATION MANAGEMENT ####

import uuid
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
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
    - metadata: *Optional[dict]* - Metadata for team, store information for team. Example metadata - {"extra_info": "some info"}
    - blocked: *bool* - Flag indicating if the org is blocked or not - will stop all calls from keys with this org_id.
    - tags: *Optional[List[str]]* - Tags for [tracking spend](https://litellm.vercel.app/docs/proxy/enterprise#tracking-spend-for-custom-tags) and/or doing [tag-based routing](https://litellm.vercel.app/docs/proxy/tag_routing).
    - organization_id: *Optional[str]* - The organization id of the team. Default is None. Create via `/organization/new`.
    - model_aliases: Optional[dict] - Model aliases for the team. [Docs](https://docs.litellm.ai/docs/proxy/team_based_routing#create-team-with-model-alias)


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
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

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
            if m not in user_api_key_dict.models:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"User not allowed to give access to model={m}. Models you have access to = {user_api_key_dict.models}"
                    },
                )
    organization_row = LiteLLM_OrganizationTable(
        **data.json(exclude_none=True),
        created_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
        updated_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
    )
    new_organization_row = prisma_client.jsonify_object(
        organization_row.json(exclude_none=True)
    )
    response = await prisma_client.db.litellm_organizationtable.create(
        data={
            **new_organization_row,  # type: ignore
        }
    )

    return response


@router.post(
    "/organization/update",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_organization():
    """[TODO] Not Implemented yet. Let us know if you need this - https://github.com/BerriAI/litellm/issues"""
    raise NotImplementedError("Not Implemented Yet")


@router.post(
    "/organization/delete",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_organization():
    """[TODO] Not Implemented yet. Let us know if you need this - https://github.com/BerriAI/litellm/issues"""
    raise NotImplementedError("Not Implemented Yet")


@router.get(
    "/organization/list",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
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

    if (
        user_api_key_dict.user_role is None
        or user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": f"Only admins can list orgs. Your role is = {user_api_key_dict.user_role}"
            },
        )
    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    response = await prisma_client.db.litellm_organizationtable.find_many()

    return response


@router.post(
    "/organization/info",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def info_organization(data: OrganizationRequest):
    """
    Get the org specific information
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
            updated_user, updated_organization_membership = (
                await add_member_to_organization(
                    member=member,
                    organization_id=data.organization_id,
                    prisma_client=prisma_client,
                )
            )

            updated_users.append(updated_user)
            updated_organization_memberships.append(updated_organization_membership)

        return OrganizationAddMemberResponse(
            organization_id=data.organization_id,
            updated_users=updated_users,
            updated_organization_memberships=updated_organization_memberships,
        )
    except Exception as e:
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

        if member.user_email is not None:
            existing_user_email_row = (
                await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_email": member.user_email}
                )
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
        raise ValueError(f"Error adding member to organization: {e}")
