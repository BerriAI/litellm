from datetime import date
from functools import partial
from math import ceil
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Query, Request

from litellm.proxy.list_api.common import (
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    build_list_links,
    reject_unknown_query_params,
)
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import (
    raise_public,
    reject_duplicate_list_params,
    require_customer_admin,
)
from litellm.proxy.management_helpers.customer_hierarchy import ErrorCode, Failure
from litellm.proxy.utils import query_raw_rows
from litellm.repositories.customer_usage_repository import CustomerUsageRepository
from litellm.types.proxy.management_endpoints.customer_usage import CustomerOrganizationUsage
from litellm.types.proxy.management_endpoints.management_v1 import ListMeta, ListResponse, ProblemDetail


def get_customer_usage_repository() -> CustomerUsageRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise_public(Failure(ErrorCode.DATABASE_UNAVAILABLE))
    return CustomerUsageRepository(partial(query_raw_rows, prisma_client.db))


router: Final = APIRouter(
    prefix=MANAGEMENT_V1_PREFIX,
    tags=["customer hierarchy"],
    dependencies=(
        Depends(require_customer_admin),
        Depends(reject_unknown_query_params),
        Depends(reject_duplicate_list_params),
    ),
)


@router.get(
    "/customer_organization_usage",
    response_model=ListResponse[CustomerOrganizationUsage],
    description=(
        "List customer organizations with spend, requests and total input plus output tokens for an inclusive "
        "UTC date range. Usage matches the request user (end-user ID) to customer_member_id and rolls up through "
        "the member's current team and organization. Organizations without usage return zero totals. "
        "Only retained daily usage for currently registered members is included. Proxy admins only."
    ),
)
async def list_customer_organization_usage(
    request: Request,
    repository: Annotated[CustomerUsageRepository, Depends(get_customer_usage_repository)],
    start_date: Annotated[date, Query(description="First UTC calendar day, inclusive")],
    end_date: Annotated[date, Query(description="Last UTC calendar day, inclusive")],
    q: Annotated[str, Query(description="Case-insensitive substring of customer organization ID or alias")] = "",
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Page size")] = 50,
) -> ListResponse[CustomerOrganizationUsage]:
    if start_date > end_date:
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}invalid-date-range",
                title="Invalid date range",
                status=400,
                detail="start_date must be on or before end_date.",
            )
        )
    result: Final = await repository.list_organizations(start_date, end_date, q, page, page_size)
    total_pages: Final = ceil(result.total_count / page_size)
    return ListResponse(
        data=list(result.rows),
        meta=ListMeta(total_count=result.total_count, page=page, page_size=page_size, total_pages=total_pages),
        links=build_list_links(request, page, total_pages),
    )
