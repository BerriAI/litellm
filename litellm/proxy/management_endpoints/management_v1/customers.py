from typing import Annotated, Final, NoReturn

from fastapi import APIRouter, Depends

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem, reject_unknown_query_params
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchy,
    CustomerMember,
    Entity,
    Failure,
    Kind,
    MemberCreate,
    Organization,
    OrganizationCreate,
    OrganizationPatch,
    Team,
    TeamCreate,
    TeamPatch,
)
from litellm.proxy.management_helpers.utils import (
    management_endpoint_wrapper,  # pyright: ignore[reportUnknownVariableType]  # legacy decorator is untyped
)
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


def raise_public(failure: Failure) -> NoReturn:
    raise ManagementProblem(
        ProblemDetail(
            type=f"{PROBLEM_TYPE_BASE}customer-hierarchy-{failure.status}",
            title={
                403: "Forbidden",
                404: "Not found",
                409: "Conflict",
                422: "Invalid budget template",
                503: "Database not connected",
            }.get(failure.status, "Internal server error"),
            status=failure.status,
            detail=failure.detail,
        )
    )


def require_customer_admin(
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise_public(Failure(403, "Only proxy admins can manage customer hierarchies."))
    return auth


def get_customer_hierarchy() -> CustomerHierarchy:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise_public(Failure(503, "Database not connected."))
    return CustomerHierarchy(prisma_client)


def resolve(result: Entity | Failure) -> Entity:
    if isinstance(result, Failure):
        raise_public(result)
    return result


@router.post(
    "/customer_organizations",
    tags=["customer hierarchy"],
    dependencies=(Depends(reject_unknown_query_params),),
    response_model=Organization,
    description="Create a customer organization and its resolved budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def post_customer_organizations(
    data: OrganizationCreate,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Entity:
    return resolve(await hierarchy.write(Kind.ORGANIZATION, data, user_api_key_dict.user_id or "proxy_admin"))


@router.get(
    "/customer_organizations/{customer_org_id}",
    tags=["customer hierarchy"],
    dependencies=(Depends(reject_unknown_query_params),),
    response_model=Organization,
    description="Get a customer organization and its resolved budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def get_customer_organizations(
    customer_org_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Entity:
    return resolve(await hierarchy.get(Kind.ORGANIZATION, customer_org_id))


@router.patch(
    "/customer_organizations/{customer_org_id}",
    tags=["customer hierarchy"],
    dependencies=(Depends(reject_unknown_query_params),),
    response_model=Organization,
    description="Update a customer organization and its resolved budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def patch_customer_organizations(
    customer_org_id: str,
    data: OrganizationPatch,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Entity:
    return resolve(
        await hierarchy.write(Kind.ORGANIZATION, data, user_api_key_dict.user_id or "proxy_admin", customer_org_id)
    )


@router.post(
    "/customer_teams",
    tags=["customer hierarchy"],
    dependencies=(Depends(reject_unknown_query_params),),
    response_model=Team,
    description="Create a customer team and its resolved budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def post_customer_teams(
    data: TeamCreate,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Entity:
    return resolve(await hierarchy.write(Kind.TEAM, data, user_api_key_dict.user_id or "proxy_admin"))


@router.get(
    "/customer_teams/{customer_team_id}",
    tags=["customer hierarchy"],
    dependencies=(Depends(reject_unknown_query_params),),
    response_model=Team,
    description="Get a customer team and its resolved budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def get_customer_teams(
    customer_team_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Entity:
    return resolve(await hierarchy.get(Kind.TEAM, customer_team_id))


@router.patch(
    "/customer_teams/{customer_team_id}",
    tags=["customer hierarchy"],
    dependencies=(Depends(reject_unknown_query_params),),
    response_model=Team,
    description="Update a customer team and its resolved budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def patch_customer_teams(
    customer_team_id: str,
    data: TeamPatch,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Entity:
    return resolve(await hierarchy.write(Kind.TEAM, data, user_api_key_dict.user_id or "proxy_admin", customer_team_id))


@router.post(
    "/customer_members",
    tags=["customer hierarchy"],
    dependencies=(Depends(reject_unknown_query_params),),
    response_model=CustomerMember,
    description="Create a customer member and its resolved budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def post_customer_members(
    data: MemberCreate,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Entity:
    return resolve(await hierarchy.write(Kind.MEMBER, data, user_api_key_dict.user_id or "proxy_admin"))


@router.get(
    "/customer_members/{customer_member_id}",
    tags=["customer hierarchy"],
    dependencies=(Depends(reject_unknown_query_params),),
    response_model=CustomerMember,
    description="Get a customer member and its resolved budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def get_customer_members(
    customer_member_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Entity:
    return resolve(await hierarchy.get(Kind.MEMBER, customer_member_id))


@router.patch(
    "/customer_members/{customer_member_id}",
    tags=["customer hierarchy"],
    dependencies=(Depends(reject_unknown_query_params),),
    response_model=CustomerMember,
    description="Update a customer member and its resolved budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def patch_customer_members(
    customer_member_id: str,
    data: TeamPatch,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Entity:
    return resolve(
        await hierarchy.write(Kind.MEMBER, data, user_api_key_dict.user_id or "proxy_admin", customer_member_id)
    )
