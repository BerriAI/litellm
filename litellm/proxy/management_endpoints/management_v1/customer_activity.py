from collections.abc import Mapping, Sequence
from functools import partial
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from litellm.proxy.management_endpoints.common_daily_activity import (
    DailySpendRecord,
    InvalidDateRange,
    daily_activity_repository,
    daily_activity_scope,
    get_daily_activity,
    get_daily_activity_aggregated,
)
from litellm.proxy.management_endpoints.common_daily_activity import (
    raise_public as raise_activity_error,
)
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import (
    raise_public,
    reject_duplicate_list_params,
    require_customer_admin,
)
from litellm.proxy.management_endpoints.team_endpoints import aggregated_date_range_error
from litellm.proxy.management_helpers.customer_hierarchy import ErrorCode, Failure
from litellm.proxy.utils import PrismaClient, query_raw_rows
from litellm.repositories.customer_activity_repository import CustomerActivityRepository, CustomerEntity
from litellm.types.proxy.management_endpoints.common_daily_activity import SpendAnalyticsPaginatedResponse


class ActivityQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_date: str | None = Field(default=None, description="First UTC calendar day, YYYY-MM-DD, inclusive")
    end_date: str | None = Field(default=None, description="Last UTC calendar day, YYYY-MM-DD, inclusive")
    model: str | None = None
    api_key: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1)


class OrganizationActivityQuery(ActivityQuery):
    customer_org_ids: str | None = Field(default=None, description="Comma-separated customer organization IDs")
    exclude_customer_org_ids: str | None = None


class TeamActivityQuery(ActivityQuery):
    customer_team_ids: str | None = Field(default=None, description="Comma-separated customer team IDs")
    exclude_customer_team_ids: str | None = None


class MemberActivityQuery(ActivityQuery):
    customer_member_ids: str | None = Field(default=None, description="Comma-separated customer member IDs")
    exclude_customer_member_ids: str | None = None


class OrganizationAggregatedQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_date: str = Field(description="First UTC calendar day, YYYY-MM-DD, inclusive")
    end_date: str = Field(description="Last UTC calendar day, YYYY-MM-DD, inclusive, at most 400 days after start")
    customer_org_ids: str = Field(description="Comma-separated IDs for up to 100 visible customer organizations")
    model: str | None = None
    api_key: str | None = None

    @field_validator("customer_org_ids")
    @classmethod
    def bounded_organizations(cls, value: str) -> str:
        ids: Final = split_ids(value)
        if not ids or len(ids) > 100:
            raise ValueError("Provide between 1 and 100 customer organization IDs")
        return value


_ID: Final = TypeAdapter(str)


def get_customer_activity_client() -> PrismaClient:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise_public(Failure(ErrorCode.DATABASE_UNAVAILABLE))
    return prisma_client


def split_ids(value: str | None) -> tuple[str, ...] | None:
    return tuple(part.strip() for part in value.split(",") if part.strip()) if value is not None else None


async def customer_activity(
    client: PrismaClient,
    entity: CustomerEntity,
    query: ActivityQuery,
    ids: str | None,
    excluded_ids: str | None,
) -> SpendAnalyticsPaginatedResponse:
    repository: Final = CustomerActivityRepository(partial(query_raw_rows, client.db))

    async def aliases(records: Sequence[DailySpendRecord]) -> Mapping[str, Mapping[str, object]]:
        resolved_ids: Final = tuple(sorted({_ID.validate_python(getattr(row, entity.value)) for row in records}))
        return await repository.aliases(entity, resolved_ids)

    return await get_daily_activity(
        prisma_client=client,
        table_name="litellm_dailycustomermemberspend",
        entity_id_field=entity.value,
        entity_id=split_ids(ids),
        exclude_entity_ids=split_ids(excluded_ids),
        entity_metadata_field={},
        resolve_entity_metadata=aliases,
        start_date=query.start_date,
        end_date=query.end_date,
        model=query.model,
        api_key=query.api_key,
        page=query.page,
        page_size=query.page_size,
    )


router: Final = APIRouter(
    prefix=MANAGEMENT_V1_PREFIX,
    tags=["customer hierarchy"],
    dependencies=(
        Depends(require_customer_admin),
        Depends(reject_duplicate_list_params),
    ),
)


@router.get(
    "/customer_organizations/daily/activity",
    response_model=SpendAnalyticsPaginatedResponse,
    description=(
        "Get daily customer organization activity for inclusive UTC dates, using the recorded organization IDs "
        "on customer member daily spend rows. Includes standard metrics and breakdowns with organization aliases. "
        "Dates, model, API key, entity and exclusion filters apply before pagination. Proxy admins only."
    ),
)
async def get_customer_organization_daily_activity(
    query: Annotated[OrganizationActivityQuery, Query()],
    client: Annotated[PrismaClient, Depends(get_customer_activity_client)],
) -> SpendAnalyticsPaginatedResponse:
    return await customer_activity(
        client, CustomerEntity.ORGANIZATION, query, query.customer_org_ids, query.exclude_customer_org_ids
    )


@router.get(
    "/customer_teams/daily/activity",
    response_model=SpendAnalyticsPaginatedResponse,
    description=(
        "Get daily customer team activity for inclusive UTC dates, using the recorded team IDs on customer member "
        "daily spend rows. Includes standard metrics and breakdowns with team aliases. Dates, model, API key, entity "
        "and exclusion filters apply before pagination. Proxy admins only."
    ),
)
async def get_customer_team_daily_activity(
    query: Annotated[TeamActivityQuery, Query()],
    client: Annotated[PrismaClient, Depends(get_customer_activity_client)],
) -> SpendAnalyticsPaginatedResponse:
    return await customer_activity(
        client, CustomerEntity.TEAM, query, query.customer_team_ids, query.exclude_customer_team_ids
    )


@router.get(
    "/customer_members/daily/activity",
    response_model=SpendAnalyticsPaginatedResponse,
    description=(
        "Get daily customer member activity for inclusive UTC dates. Includes standard metrics and breakdowns "
        "with member aliases. Dates, model, API key, entity and exclusion filters apply before pagination. "
        "Member identity is supplied by x-litellm-customer-member-id when the writer records usage. Proxy admins only."
    ),
)
async def get_customer_member_daily_activity(
    query: Annotated[MemberActivityQuery, Query()],
    client: Annotated[PrismaClient, Depends(get_customer_activity_client)],
) -> SpendAnalyticsPaginatedResponse:
    return await customer_activity(
        client, CustomerEntity.MEMBER, query, query.customer_member_ids, query.exclude_customer_member_ids
    )


@router.get(
    "/customer_organizations/daily/activity/aggregated",
    response_model=SpendAnalyticsPaginatedResponse,
    description=(
        "Get complete daily usage totals for up to 100 customer organizations in one response. "
        "Uses the shared SQL daily-spend aggregation, including per-organization breakdowns. "
        "Dates are inclusive UTC days, limited to 400 days, and model/API-key filters apply before aggregation. "
        "Key breakdowns show the top key; totals include every key. No raw-row pagination is required. "
        "Budget counters are returned separately by organization reads. "
        "Proxy admins only."
    ),
)
async def get_customer_organization_daily_activity_aggregated(
    query: Annotated[OrganizationAggregatedQuery, Query()],
    client: Annotated[PrismaClient, Depends(get_customer_activity_client)],
) -> SpendAnalyticsPaginatedResponse:
    date_error: Final = aggregated_date_range_error(query.start_date, query.end_date)
    if date_error is not None:
        raise_activity_error(InvalidDateRange(reason=date_error))
    ids: Final = split_ids(query.customer_org_ids)
    scope: Final = daily_activity_scope(
        "litellm_dailycustomermemberspend",
        "customer_org_id",
        ids,
        None,
        query.api_key,
        query.start_date,
        query.end_date,
        query.model,
        None,
    )
    aliases: Final = await CustomerActivityRepository(partial(query_raw_rows, client.db)).aliases(
        CustomerEntity.ORGANIZATION, tuple(ids or ())
    )
    return await get_daily_activity_aggregated(
        daily_activity_repository(client),
        scope,
        entity_metadata_field=aliases,
        include_entity_breakdown=True,
        api_key_limit=1,
    )
