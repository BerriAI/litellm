from collections.abc import Mapping
from math import ceil
from types import MappingProxyType
from typing import Annotated, Final, NoReturn, TypeAlias, TypeVar

from fastapi import APIRouter, Depends, Query, Request, Response

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.customer_hierarchy_auth import forget_customer_hierarchies, forget_customer_organization_aliases
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.list_api.common import (
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    build_list_links,
    reject_unknown_query_params,
)
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchy,
    CustomerPortalLimits,
    ErrorCode,
    Failure,
    Page,
)
from litellm.proxy.management_helpers.customer_upsert import CustomerUpsert
from litellm.proxy.management_helpers.utils import (
    management_endpoint_wrapper,  # pyright: ignore[reportUnknownVariableType]  # legacy decorator is untyped
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    BudgetGroupsUpdate,
    BudgetTemplate,
    CustomerMember,
    MemberCreate,
    MemberPatch,
    MemberUpsert,
    MemberUpsertResponse,
    Organization,
    OrganizationCreate,
    OrganizationPatch,
    OrganizationUpsert,
    OrganizationUpsertResponse,
    Team,
    TeamCreate,
    TeamPatch,
    TemplatePatch,
)
from litellm.types.proxy.management_endpoints.management_v1 import ListMeta, ListResponse, ProblemDetail

T: Final = TypeVar("T")


_ERROR_DETAILS: Final[Mapping[ErrorCode, tuple[int, str, str]]] = MappingProxyType(
    {
        ErrorCode.CAPACITY_BUSY: (
            503,
            "Customer portal busy",
            "Customer portal capacity is being updated. Retry the request.",
        ),
        ErrorCode.FREE_TIER_LIMIT: (
            403,
            "Free customer portal limit reached",
            "Free customer portal limit reached: up to 5 customer organizations, 5 teams per organization and "
            "5 members per team without a LiteLLM Enterprise license. Upgrade to Enterprise to remove the limits: "
            "https://www.litellm.ai/enterprise",
        ),
        ErrorCode.NOT_FOUND: (404, "Not found", "Customer entity not found."),
        ErrorCode.PARENT_NOT_FOUND: (404, "Parent not found", "Customer parent not found."),
        ErrorCode.ALREADY_EXISTS: (409, "Already exists", "Customer entity ID already exists."),
        ErrorCode.INVALID_DEFAULT_BUDGET: (
            422,
            "Invalid default budget",
            "default_budget_template must name one of default_budgets, or default_budgets must hold one template.",
        ),
        ErrorCode.TEMPLATE_NOT_IN_ORG: (
            422,
            "Template not in organization",
            "Budget template must belong to the member's organization.",
        ),
        ErrorCode.HAS_CHILDREN: (409, "Has children", "Delete the customer's children before deleting the parent."),
        ErrorCode.INVALID_TEMPLATE_NAMES: (
            422,
            "Invalid template names",
            "Budget template names must be unique within the organization.",
        ),
        ErrorCode.TEMPLATE_IN_USE: (409, "Template in use", "A default or referenced template cannot be deleted."),
        ErrorCode.BUDGET_REQUIRED: (422, "Budget required", "A member requires a template or a budget override."),
        ErrorCode.INCREASE_NOT_ALLOWED: (
            422,
            "Increase not allowed",
            "The member's budget does not allow temporary increase requests.",
        ),
        ErrorCode.REQUEST_NOT_PENDING: (
            409,
            "Request not pending",
            "Only a pending increase request can be approved or rejected.",
        ),
        ErrorCode.REQUEST_ALREADY_PENDING: (
            409,
            "Request already pending",
            "The member already has an increase request waiting for review.",
        ),
        ErrorCode.EXPIRY_IN_PAST: (422, "Expiry in past", "A temporary increase must expire in the future."),
        ErrorCode.KEY_NOT_IN_ORG: (
            403,
            "Key not in organization",
            "The key may only act for members of its own customer organization.",
        ),
        ErrorCode.FORBIDDEN: (403, "Forbidden", "Only proxy admins can manage customer hierarchies."),
        ErrorCode.DATABASE_UNAVAILABLE: (503, "Database not connected", "Database not connected."),
        ErrorCode.CONFIGURATION_CONFLICT: (
            409,
            "Budget groups changed",
            "Budget groups changed since you started editing. Reload before saving.",
        ),
        ErrorCode.ALIAS_ALREADY_EXISTS: (
            409,
            "Customer org alias already exists",
            "Customer org alias is already used by another customer org.",
        ),
        ErrorCode.CUSTOMER_ORG_MISMATCH: (
            409,
            "Customer org mismatch",
            "Customer member and customer team must belong to the specified customer org.",
        ),
    }
)
assert _ERROR_DETAILS.keys() == frozenset(ErrorCode), "every ErrorCode needs a public problem"


def error_details(code: ErrorCode) -> tuple[int, str, str]:
    return _ERROR_DETAILS[code]


def raise_public(failure: Failure) -> NoReturn:
    status, title, detail = error_details(failure.code)
    raise ManagementProblem(
        ProblemDetail(
            type=f"{PROBLEM_TYPE_BASE}customer-hierarchy-{failure.code.value.replace('_', '-')}",
            title=title,
            status=status,
            detail=detail,
        )
    )


def require_customer_admin(auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> UserAPIKeyAuth:
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise_public(Failure(ErrorCode.FORBIDDEN))
    return auth


def customer_actor(auth: Annotated[UserAPIKeyAuth, Depends(require_customer_admin)]) -> str:
    return auth.user_id or "proxy_admin"


def get_customer_hierarchy() -> CustomerHierarchy:
    from litellm.proxy.proxy_server import premium_user, prisma_client

    if prisma_client is None:
        raise_public(Failure(ErrorCode.DATABASE_UNAVAILABLE))
    return CustomerHierarchy(prisma_client, premium_user=premium_user)


def resolve(result: T | Failure) -> T:
    if isinstance(result, Failure):
        raise_public(result)
    return result


def list_response(result: Page[T], request: Request, page: int, page_size: int) -> ListResponse[T]:
    total_pages: Final = ceil(result.total_count / page_size)
    return ListResponse[T].model_validate(
        {
            "data": result.data,
            "meta": ListMeta(total_count=result.total_count, page=page, page_size=page_size, total_pages=total_pages),
            "links": build_list_links(request, page, total_pages),
        }
    )


def reject_duplicate_list_params(request: Request) -> None:
    repeated: Final = tuple(
        sorted(name for name in request.query_params if len(request.query_params.getlist(name)) > 1)
    )
    if repeated:
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}duplicate-query-parameter",
                title="Duplicate query parameter",
                status=400,
                detail=f"Repeated query parameter(s): {', '.join(repeated)}. Each may appear once.",
            )
        )


router: Final = APIRouter(
    prefix=MANAGEMENT_V1_PREFIX,
    tags=["customer hierarchy"],
    dependencies=(Depends(require_customer_admin), Depends(reject_unknown_query_params)),
)


def get_user_api_key_cache() -> UserApiKeyCache:
    from litellm.proxy.proxy_server import user_api_key_cache

    return user_api_key_cache


AuthCache: TypeAlias = Annotated[UserApiKeyCache, Depends(get_user_api_key_cache)]


@router.get(
    "/customer_portal/limits",
    response_model=CustomerPortalLimits,
    description="Get customer portal free-tier limits and current capacity. Proxy admins only.",
)
@management_endpoint_wrapper
async def get_customer_portal_limits(
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> CustomerPortalLimits:
    return resolve(await hierarchy.portal_limits())


@router.post(
    "/customer_organizations",
    status_code=201,
    response_model=Organization,
    description="Create a customer organization with resolved budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def create_customer_organization(
    data: OrganizationCreate,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Organization:
    return resolve(await hierarchy.create_organization(data, actor))


@router.post(
    "/customer_organizations/upsert",
    response_model=OrganizationUpsertResponse,
    description=(
        "Create or update a customer org by customer_org_id, or by alias when no ID is supplied. "
        "Returns 200 with created=true for creation and created=false for updates. Repeated identical calls are no-ops. "
        "Supplied default_budgets are merged by template name, preserving existing template IDs and omitted templates. "
        "A single supplied template becomes the default; multiple templates require default_budget_template to name one. "
        "Omitted fields retain their values. Conflicting or ambiguous aliases return 409. Proxy admins only."
    ),
)
@management_endpoint_wrapper
async def upsert_customer_organization(
    data: OrganizationUpsert,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cache: AuthCache,
) -> OrganizationUpsertResponse:
    outcome: Final = resolve(await CustomerUpsert(hierarchy).organization(data, actor))
    await forget_customer_hierarchies(
        resolve(await hierarchy.member_ids_in_organization(outcome.data.customer_org_id)), cache
    )
    await forget_customer_organization_aliases((outcome.previous_alias, outcome.data.alias), cache)
    return outcome.data


@router.get(
    "/customer_organizations/{customer_org_id}",
    response_model=Organization,
    description="Get a customer organization with resolved budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def get_customer_organization(
    customer_org_id: str,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Organization:
    return resolve(await hierarchy.get_organization(customer_org_id))


@router.patch(
    "/customer_organizations/{customer_org_id}",
    response_model=Organization,
    description="Update a customer organization alias or overall budget. Proxy admins only.",
)
@management_endpoint_wrapper
async def update_customer_organization(
    customer_org_id: str,
    data: OrganizationPatch,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cache: AuthCache,
) -> Organization:
    previous: Final = resolve(await hierarchy.get_organization(customer_org_id))
    organization: Final = resolve(await hierarchy.update_organization(customer_org_id, data, actor))
    await forget_customer_hierarchies(resolve(await hierarchy.member_ids_in_organization(customer_org_id)), cache)
    await forget_customer_organization_aliases((previous.alias, organization.alias), cache)
    return organization


@router.delete(
    "/customer_organizations/{customer_org_id}",
    status_code=204,
    description="Delete a customer organization and its owned budgets. Set cascade=true to also delete its customer teams and members; otherwise parents with children return 409. Proxy admins only.",
)
@management_endpoint_wrapper
async def delete_customer_organization(
    customer_org_id: str,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cache: AuthCache,
    cascade: Annotated[bool, Query(description="Also delete this customer organization's teams and members.")] = False,
) -> Response:
    previous: Final = resolve(await hierarchy.get_organization(customer_org_id))
    resolve(await hierarchy.delete_organization(customer_org_id, cascade=cascade))
    await forget_customer_organization_aliases((previous.alias,), cache)
    return Response(status_code=204)


@router.get(
    "/customer_organizations",
    response_model=ListResponse[Organization],
    dependencies=(Depends(reject_duplicate_list_params),),
    description="List customer organizations with page-number pagination and resolved budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def list_customer_organizations(
    request: Request,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1)] = 50,
    q: Annotated[str, Query(description="Case-insensitive substring of the customer ID or alias")] = "",
) -> ListResponse[Organization]:
    size: Final = min(page_size, 100)
    result: Final = resolve(await hierarchy.list_organizations(page, size, q.strip()))
    return list_response(result, request, page, size)


@router.post(
    "/customer_teams",
    status_code=201,
    response_model=Team,
    description="Create a customer team with resolved budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def create_customer_team(
    data: TeamCreate,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Team:
    return resolve(await hierarchy.create_team(data, actor))


@router.get(
    "/customer_teams/{customer_team_id}",
    response_model=Team,
    description="Get a customer team with resolved budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def get_customer_team(
    customer_team_id: str,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Team:
    return resolve(await hierarchy.get_team(customer_team_id))


@router.patch(
    "/customer_teams/{customer_team_id}",
    response_model=Team,
    description="Update a customer team and its budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def update_customer_team(
    customer_team_id: str,
    data: TeamPatch,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cache: AuthCache,
) -> Team:
    team: Final = resolve(await hierarchy.update_team(customer_team_id, data, actor))
    await forget_customer_hierarchies(resolve(await hierarchy.member_ids_in_team(customer_team_id)), cache)
    return team


@router.delete(
    "/customer_teams/{customer_team_id}",
    status_code=204,
    description="Delete a customer team and its owned budgets. Set cascade=true to also delete its customer members; otherwise parents with children return 409. Proxy admins only.",
)
@management_endpoint_wrapper
async def delete_customer_team(
    customer_team_id: str,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cascade: Annotated[bool, Query(description="Also delete this customer team's members.")] = False,
) -> Response:
    resolve(await hierarchy.delete_team(customer_team_id, cascade=cascade))
    return Response(status_code=204)


@router.get(
    "/customer_teams",
    response_model=ListResponse[Team],
    dependencies=(Depends(reject_duplicate_list_params),),
    description="List customer teams with page-number pagination and resolved budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def list_customer_teams(
    request: Request,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1)] = 50,
    parent_org_id: str | None = None,
) -> ListResponse[Team]:
    size: Final = min(page_size, 100)
    result: Final = resolve(await hierarchy.list_teams(parent_org_id, page, size))
    return list_response(result, request, page, size)


@router.post(
    "/customer_members",
    status_code=201,
    response_model=CustomerMember,
    description="Create a customer member with resolved budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def create_customer_member(
    data: MemberCreate,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> CustomerMember:
    return resolve(await hierarchy.create_member(data, actor))


@router.post(
    "/customer_members/upsert",
    response_model=MemberUpsertResponse,
    description=(
        "Create or update a customer member by customer_member_id within a customer org identified by "
        "customer_org_id or customer_org_alias. If both are supplied they must match. "
        "A new customer member with no parent_team_id joins the customer org's default customer team, created if necessary. An existing customer member keeps its customer teams, and a supplied parent_team_id becomes its first customer team. "
        "Alias, budget_id, budget_override and budget_override_expires_at follow the existing customer member models. "
        "Omitted update fields retain their values. Moving between customer teams in the same customer org is allowed. "
        "Moving between customer orgs or selecting another customer org's customer team returns 409. "
        "Returns 200 with created=true for creation and created=false for updates. Repeated identical calls are no-ops. "
        "Proxy admins only."
    ),
)
@management_endpoint_wrapper
async def upsert_customer_member(
    data: MemberUpsert,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cache: AuthCache,
) -> MemberUpsertResponse:
    member: Final = resolve(await CustomerUpsert(hierarchy).member(data, actor))
    await forget_customer_hierarchies((member.customer_member_id,), cache)
    return member


@router.get(
    "/customer_members/{customer_member_id}",
    response_model=CustomerMember,
    description="Get a customer member with resolved budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def get_customer_member(
    customer_member_id: str,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> CustomerMember:
    return resolve(await hierarchy.get_member(customer_member_id))


@router.patch(
    "/customer_members/{customer_member_id}",
    response_model=CustomerMember,
    description="Update a customer member and its budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def update_customer_member(
    customer_member_id: str,
    data: MemberPatch,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cache: AuthCache,
) -> CustomerMember:
    member: Final = resolve(await hierarchy.update_member(customer_member_id, data, actor))
    await forget_customer_hierarchies((customer_member_id,), cache)
    return member


@router.delete(
    "/customer_members/{customer_member_id}",
    status_code=204,
    description="Delete a customer member and its owned budgets. Parents with children return 409. Proxy admins only.",
)
@management_endpoint_wrapper
async def delete_customer_member(
    customer_member_id: str,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cache: AuthCache,
) -> Response:
    resolve(await hierarchy.delete_member(customer_member_id))
    await forget_customer_hierarchies((customer_member_id,), cache)
    return Response(status_code=204)


@router.get(
    "/customer_members",
    response_model=ListResponse[CustomerMember],
    dependencies=(Depends(reject_duplicate_list_params),),
    description="List customer members with page-number pagination and resolved budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def list_customer_members(
    request: Request,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1)] = 50,
    parent_team_id: str | None = None,
    parent_org_id: str | None = None,
) -> ListResponse[CustomerMember]:
    size: Final = min(page_size, 100)
    result: Final = resolve(await hierarchy.list_members(parent_team_id, page, size, parent_org_id=parent_org_id))
    return list_response(result, request, page, size)


@router.post(
    "/customer_organizations/{customer_org_id}/default_budgets",
    status_code=201,
    response_model=Organization,
    description="Add a named budget template to a customer organization. Proxy admins only.",
)
@management_endpoint_wrapper
async def add_customer_budget_template(
    customer_org_id: str,
    data: BudgetTemplate,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Organization:
    return resolve(await hierarchy.add_template(customer_org_id, data, actor))


@router.patch(
    "/customer_organizations/{customer_org_id}/default_budgets",
    response_model=Organization,
    description="Atomically configure budget groups and their default. Requires the current budget_groups_revision. "
    "Existing groups retain omitted settings and window reset dates. Stale revisions and referenced deletions "
    "return 409 without applying changes. Proxy admins only.",
)
@management_endpoint_wrapper
async def configure_customer_budget_groups(
    customer_org_id: str,
    data: BudgetGroupsUpdate,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cache: AuthCache,
) -> Organization:
    organization: Final = resolve(await hierarchy.configure_templates(customer_org_id, data, actor))
    await forget_customer_hierarchies(resolve(await hierarchy.member_ids_in_organization(customer_org_id)), cache)
    return organization


@router.patch(
    "/customer_organizations/{customer_org_id}/default_budgets/{budget_id}",
    response_model=Organization,
    description="Edit a budget template in place. Members inherit the updated values. Proxy admins only.",
)
@management_endpoint_wrapper
async def update_customer_budget_template(
    customer_org_id: str,
    budget_id: str,
    data: TemplatePatch,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
    cache: AuthCache,
) -> Organization:
    organization: Final = resolve(await hierarchy.update_template(customer_org_id, budget_id, data, actor))
    await forget_customer_hierarchies(resolve(await hierarchy.member_ids_using_template(budget_id)), cache)
    return organization


@router.delete(
    "/customer_organizations/{customer_org_id}/default_budgets/{budget_id}",
    status_code=204,
    description="Delete an unreferenced, non-default customer budget template. Proxy admins only.",
)
@management_endpoint_wrapper
async def delete_customer_budget_template(
    customer_org_id: str,
    budget_id: str,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> Response:
    resolve(await hierarchy.delete_template(customer_org_id, budget_id, actor))
    return Response(status_code=204)
