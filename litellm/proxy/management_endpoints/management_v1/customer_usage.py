from collections.abc import Sequence
from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Annotated, Final, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AfterValidator, TypeAdapter

from litellm.litellm_core_utils.core_helpers import require_attributes
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem, reject_unknown_query_params
from litellm.proxy.management_endpoints.common_daily_activity import (
    DailySpendRecord,
    InvalidDateRange,
    daily_activity_repository,
    daily_activity_scope,
    get_daily_activity,
    get_daily_activity_aggregated,
)
from litellm.proxy.management_endpoints.common_daily_activity import raise_public as raise_activity_error
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import (
    raise_public,
    reject_duplicate_list_params,
)
from litellm.proxy.management_endpoints.team_endpoints import aggregated_date_range_error
from litellm.proxy.management_helpers.app_access import granted_app_ids
from litellm.proxy.management_helpers.app_groups import AppGroupRepository
from litellm.proxy.management_helpers.apps import AppRepository
from litellm.proxy.management_helpers.customer_hierarchy import ErrorCode, Failure, PrismaRecord
from litellm.proxy.management_helpers.utils import (
    management_endpoint_wrapper,  # pyright: ignore[reportUnknownVariableType]  # legacy decorator is untyped
)
from litellm.proxy.utils import PrismaClient
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.prisma_protocols import DatabaseClient, TableActions
from litellm.types.proxy.management_endpoints.common_daily_activity import SpendAnalyticsPaginatedResponse
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail


class CustomerEntity(str, Enum):
    ORGANIZATION = "customer_org_id"
    TEAM = "customer_team_id"
    MEMBER = "customer_member_id"


class CustomerMetadataDatabase(Protocol):
    @property
    def litellm_customerorganizationtable(self) -> TableActions[PrismaRecord]: ...

    @property
    def litellm_customerteamtable(self) -> TableActions[PrismaRecord]: ...

    @property
    def litellm_customermembertable(self) -> TableActions[PrismaRecord]: ...


_TEXT: Final = TypeAdapter(str)
_OPTIONAL_TEXT: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)


def get_customer_usage_prisma() -> PrismaClient:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise_public(Failure(ErrorCode.DATABASE_UNAVAILABLE))
    return prisma_client


def metadata_table(prisma_client: PrismaClient, entity: CustomerEntity) -> TableActions[PrismaRecord]:
    database: Final = require_attributes(prisma_client, DatabaseClient, "db").db
    tables: Final = require_attributes(
        database,
        CustomerMetadataDatabase,
        "litellm_customerorganizationtable",
        "litellm_customerteamtable",
        "litellm_customermembertable",
    )
    if entity is CustomerEntity.ORGANIZATION:
        return tables.litellm_customerorganizationtable
    if entity is CustomerEntity.TEAM:
        return tables.litellm_customerteamtable
    return tables.litellm_customermembertable


async def page_aliases(
    table: TableActions[PrismaRecord], entity: CustomerEntity, records: Sequence[DailySpendRecord]
) -> dict[str, dict[str, object]]:  # mutable-ok: get_daily_activity requires dictionary-returning metadata callbacks
    identifiers: Final = tuple(_OPTIONAL_TEXT.validate_python(getattr(row, entity.value, None)) for row in records)
    return await customer_aliases(table, entity, tuple(value for value in identifiers if value))


async def customer_aliases(
    table: TableActions[PrismaRecord], entity: CustomerEntity, identifiers: Sequence[str]
) -> dict[str, dict[str, object]]:  # mutable-ok: shared daily activity helper accepts dictionary-valued metadata
    rows: Final = await find_many_in(table, entity.value, identifiers)
    values: Final = tuple(row.model_dump() for row in rows)
    return {
        _TEXT.validate_python(row[entity.value]): {
            "alias": _OPTIONAL_TEXT.validate_python(row.get("alias")) or _TEXT.validate_python(row[entity.value])
        }
        for row in values
    }


async def validate_app_filter(prisma_client: PrismaClient, app_id: str | None) -> None:
    if app_id is None:
        return
    row: Final = await AppRepository(prisma_client).table.find_unique(where={"app_id": app_id})
    if row is None or _OPTIONAL_TEXT.validate_python(row.model_dump().get("resource_id")) is None:
        raise HTTPException(status_code=404, detail="App not found.")


async def validate_group_filter(prisma_client: PrismaClient, app_group_id: str | None) -> None:
    if app_group_id is None:
        return
    row: Final = await AppGroupRepository(prisma_client).table.find_unique(where={"app_group_id": app_group_id})
    if row is None:
        raise HTTPException(status_code=404, detail="App group not found.")


async def customer_activity(
    prisma_client: PrismaClient,
    entity: CustomerEntity,
    entity_ids: str | None,
    excluded_ids: str | None,
    start_date: str | None,
    end_date: str | None,
    model: str | None,
    api_key: str | None,
    page: int,
    page_size: int,
    app_id: str | None = None,
    app_group_id: str | None = None,
) -> SpendAnalyticsPaginatedResponse:
    await validate_app_filter(prisma_client, app_id)
    await validate_group_filter(prisma_client, app_group_id)
    return await get_daily_activity(
        prisma_client=prisma_client,
        table_name="litellm_dailycustomermemberspend",
        entity_id_field=entity.value,
        entity_id=entity_ids.split(",") if entity_ids else None,
        entity_metadata_field=None,
        exclude_entity_ids=excluded_ids.split(",") if excluded_ids else None,
        start_date=start_date,
        end_date=end_date,
        model=model,
        api_key=api_key,
        page=page,
        page_size=page_size,
        app_id=app_id,
        app_group_id=app_group_id,
        resolve_entity_metadata=partial(page_aliases, metadata_table(prisma_client, entity), entity),
    )


async def require_usage_access(
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    prisma_client: Annotated[PrismaClient, Depends(get_customer_usage_prisma)],
    app_id: Annotated[str | None, Query(include_in_schema=False)] = None,
) -> None:
    """Proxy admins read all usage. Anyone else must name an app granted to them through object_permission.apps."""
    if auth.user_role == LitellmUserRoles.PROXY_ADMIN:
        return
    database: Final = require_attributes(prisma_client, DatabaseClient, "db").db
    if app_id is None or app_id not in await granted_app_ids(database, auth.user_id):
        raise_public(Failure(ErrorCode.FORBIDDEN))


router: Final = APIRouter(
    prefix=MANAGEMENT_V1_PREFIX,
    tags=["customer usage"],
    dependencies=(
        Depends(require_usage_access),
        Depends(reject_unknown_query_params),
        Depends(reject_duplicate_list_params),
    ),
)


def bounded_organizations(value: str | None) -> str | None:
    if value is None:
        return None
    ids: Final = tuple(part.strip() for part in value.split(",") if part.strip())
    if not ids or len(ids) > 100:
        raise ValueError("Provide between 1 and 100 customer organization IDs")
    return ",".join(ids)


@router.get(
    "/customer_organizations/daily/activity/aggregated",
    response_model=SpendAnalyticsPaginatedResponse,
    description=(
        "Get complete daily usage totals for up to 100 customer organizations in one response. "
        "Uses shared SQL daily-spend aggregation with per-organization breakdowns. "
        "Dates are inclusive UTC days, limited to 400 days. Model/API-key/app filters apply before aggregation. "
        "Provide either customer_org_ids or customer_org_id for one exact ID, including commas or whitespace. "
        "Key breakdowns show the top key; totals include every key. No raw-row pagination is required. "
        "Budget counters are returned separately by organization reads. Proxy admins only."
    ),
)
async def customer_organization_daily_activity_aggregated(
    prisma_client: Annotated[PrismaClient, Depends(get_customer_usage_prisma)],
    start_date: Annotated[str, Query(description="First UTC calendar day, YYYY-MM-DD, inclusive")],
    end_date: Annotated[
        str, Query(description="Last UTC calendar day, YYYY-MM-DD, inclusive, at most 400 days after start")
    ],
    customer_org_ids: Annotated[
        str | None,
        Query(description="Comma-separated IDs for up to 100 visible customer organizations"),
        AfterValidator(bounded_organizations),
    ] = None,
    customer_org_id: Annotated[
        str | None, Query(min_length=1, description="One exact customer organization ID, without splitting or trimming")
    ] = None,
    model: str | None = None,
    api_key: str | None = None,
    app_id: Annotated[
        str | None, Query(min_length=1, description="One app ID within the selected customer scope")
    ] = None,
    app_group_id: Annotated[
        str | None, Query(min_length=1, description="Group attributed when the request ran, including apps since moved")
    ] = None,
) -> SpendAnalyticsPaginatedResponse:
    await validate_app_filter(prisma_client, app_id)
    await validate_group_filter(prisma_client, app_group_id)
    date_error: Final = aggregated_date_range_error(start_date, end_date)
    if date_error is not None:
        raise_activity_error(InvalidDateRange(reason=date_error))
    if (customer_org_ids is None) == (customer_org_id is None):
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}invalid-query-parameter",
                title="Invalid organization filter",
                status=400,
                detail="Provide exactly one of customer_org_ids or customer_org_id.",
            )
        )
    ids: Final = [customer_org_id] if customer_org_id is not None else (customer_org_ids or "").split(",")
    scope: Final = daily_activity_scope(
        "litellm_dailycustomermemberspend",
        CustomerEntity.ORGANIZATION.value,
        ids,
        None,
        api_key,
        start_date,
        end_date,
        model,
        None,
    )
    aliases: Final = await customer_aliases(
        metadata_table(prisma_client, CustomerEntity.ORGANIZATION), CustomerEntity.ORGANIZATION, ids
    )
    return await get_daily_activity_aggregated(
        daily_activity_repository(prisma_client),
        replace(scope, app_id=app_id, app_group_id=app_group_id),
        entity_metadata_field=aliases,
        include_entity_breakdown=True,
        api_key_limit=1,
    )


@router.get(
    "/customer_organizations/daily/activity",
    response_model=SpendAnalyticsPaginatedResponse,
    description="Get paginated daily customer organization activity with model, provider, key and organization breakdowns. "
    "Filters accept comma-separated organization IDs. app_id narrows usage without changing global budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def customer_organization_daily_activity(
    prisma_client: Annotated[PrismaClient, Depends(get_customer_usage_prisma)],
    customer_org_ids: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    page: int = 1,
    page_size: int = 10,
    exclude_customer_org_ids: str | None = None,
    app_id: Annotated[
        str | None, Query(min_length=1, description="One app ID within the selected customer scope")
    ] = None,
    app_group_id: Annotated[
        str | None, Query(min_length=1, description="Group attributed when the request ran, including apps since moved")
    ] = None,
) -> SpendAnalyticsPaginatedResponse:
    return await customer_activity(
        prisma_client,
        CustomerEntity.ORGANIZATION,
        customer_org_ids,
        exclude_customer_org_ids,
        start_date,
        end_date,
        model,
        api_key,
        page,
        page_size,
        app_id,
        app_group_id,
    )


@router.get(
    "/customer_teams/daily/activity",
    response_model=SpendAnalyticsPaginatedResponse,
    description="Get paginated daily customer team activity with model, provider, key and team breakdowns. "
    "Filters accept comma-separated team IDs. app_id narrows usage without changing global budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def customer_team_daily_activity(
    prisma_client: Annotated[PrismaClient, Depends(get_customer_usage_prisma)],
    customer_team_ids: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    page: int = 1,
    page_size: int = 10,
    exclude_customer_team_ids: str | None = None,
    app_id: Annotated[
        str | None, Query(min_length=1, description="One app ID within the selected customer scope")
    ] = None,
    app_group_id: Annotated[
        str | None, Query(min_length=1, description="Group attributed when the request ran, including apps since moved")
    ] = None,
) -> SpendAnalyticsPaginatedResponse:
    return await customer_activity(
        prisma_client,
        CustomerEntity.TEAM,
        customer_team_ids,
        exclude_customer_team_ids,
        start_date,
        end_date,
        model,
        api_key,
        page,
        page_size,
        app_id,
        app_group_id,
    )


@router.get(
    "/customer_members/daily/activity",
    response_model=SpendAnalyticsPaginatedResponse,
    description="Get paginated daily customer member activity with model, provider, key and member breakdowns. "
    "Filters accept comma-separated member IDs. app_id narrows usage without changing global budgets. Proxy admins only.",
)
@management_endpoint_wrapper
async def customer_member_daily_activity(
    prisma_client: Annotated[PrismaClient, Depends(get_customer_usage_prisma)],
    customer_member_ids: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    page: int = 1,
    page_size: int = 10,
    exclude_customer_member_ids: str | None = None,
    app_id: Annotated[
        str | None, Query(min_length=1, description="One app ID within the selected customer scope")
    ] = None,
    app_group_id: Annotated[
        str | None, Query(min_length=1, description="Group attributed when the request ran, including apps since moved")
    ] = None,
) -> SpendAnalyticsPaginatedResponse:
    return await customer_activity(
        prisma_client,
        CustomerEntity.MEMBER,
        customer_member_ids,
        exclude_customer_member_ids,
        start_date,
        end_date,
        model,
        api_key,
        page,
        page_size,
        app_id,
        app_group_id,
    )
