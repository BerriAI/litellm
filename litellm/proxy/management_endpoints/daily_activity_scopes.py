from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Literal

from fastapi import Query

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.endpoints import resolve_agent_daily_activity_scope
from litellm.proxy.management_endpoints.common_daily_activity import ScopeDenied
from litellm.proxy.management_endpoints.customer_endpoints import (
    customer_daily_activity_is_admin,
    resolve_customer_daily_activity_scope,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import resolve_user_daily_activity_entity_ids
from litellm.proxy.management_endpoints.organization_endpoints import resolve_organization_daily_activity_scope
from litellm.proxy.management_endpoints.tag_management_endpoints import get_tag_daily_activity_api_key_filter
from litellm.proxy.management_endpoints.team_endpoints import resolve_team_daily_activity_scope
from litellm.proxy.utils import PrismaClient
from litellm.types.repositories.daily_activity import DailyActivityScope, DailyActivityTable


@dataclass(frozen=True, slots=True)
class EntityQuery:
    entity_ids: tuple[str, ...] | None
    exclude_entity_ids: tuple[str, ...]
    api_key: str | None
    start_date: str | None
    end_date: str | None
    model: str | None
    timezone_offset_minutes: int | None
    include_current_utc_day: bool
    team_ids: tuple[str, ...] | None = None
    exclude_team_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] | None = None
    exclude_tags: tuple[str, ...] = ()
    group_by: Literal["tag", "team"] | None = None


@dataclass(frozen=True, slots=True)
class ResolvedScope:
    scope: DailyActivityScope
    entity_metadata: Mapping[str, dict[str, object]] | None


Entity = Literal["user", "team", "tag", "organization", "customer", "agent"]
EntityScopeResolution = ResolvedScope | ScopeDenied
EntityScopeQuery = Callable[..., EntityQuery]
EntityScopeResolve = Callable[[UserAPIKeyAuth, EntityQuery, PrismaClient], Awaitable[EntityScopeResolution]]
OperationNames = Mapping[str, str]


@dataclass(frozen=True, slots=True)
class EntityScopeResolver:
    entity: Entity
    table: DailyActivityTable
    entity_id_field: str
    route_prefixes: tuple[str, ...]
    tags: tuple[str, ...]
    query: EntityScopeQuery
    resolve: EntityScopeResolve
    include_entity_breakdown: bool
    operation_names: OperationNames


def _query_ids(value: str | None) -> tuple[str, ...] | None:
    return tuple(value.split(",")) if value else None


def _query_excluded_ids(value: str | None) -> tuple[str, ...]:
    return tuple(value.split(",")) if value else ()


def _build_scope(
    resolver: EntityScopeResolver,
    query: EntityQuery,
    entity_ids: Sequence[str] | None,
    exclude_entity_ids: Sequence[str],
    api_key_filter: str | Sequence[str] | None,
    entity_metadata: Mapping[str, dict[str, object]] | None,
) -> ResolvedScope:
    start_date: Final[str] = query.start_date or ""
    end_date: Final[str] = query.end_date or ""
    api_keys: Final[tuple[str, ...] | None] = (
        None
        if api_key_filter is None or api_key_filter == ""
        else (api_key_filter,)
        if isinstance(api_key_filter, str)
        else tuple(api_key_filter)
    )
    return ResolvedScope(
        scope=DailyActivityScope(
            table=resolver.table,
            entity_id_field=resolver.entity_id_field,
            entity_ids=None if entity_ids is None else tuple(entity_ids),
            exclude_entity_ids=tuple(exclude_entity_ids),
            api_keys=api_keys,
            start_date=start_date,
            end_date=end_date,
            model=query.model,
            timezone_offset_minutes=query.timezone_offset_minutes,
            include_current_utc_day=query.include_current_utc_day,
        ),
        entity_metadata=entity_metadata,
    )


async def _resolve_user(
    user_api_key_dict: UserAPIKeyAuth, query: EntityQuery, prisma_client: PrismaClient
) -> EntityScopeResolution:
    entity_ids: Final = resolve_user_daily_activity_entity_ids(
        user_id=query.entity_ids[0] if query.entity_ids is not None else None,
        user_api_key_dict=user_api_key_dict,
    )
    if isinstance(entity_ids, ScopeDenied):
        return entity_ids
    return _build_scope(
        USER_RESOLVER,
        query,
        entity_ids,
        query.exclude_entity_ids,
        query.api_key,
        None,
    )


async def _resolve_team(
    user_api_key_dict: UserAPIKeyAuth, query: EntityQuery, prisma_client: PrismaClient
) -> EntityScopeResolution:
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    team_scope: Final = await resolve_team_daily_activity_scope(
        team_ids=",".join(query.entity_ids) if query.entity_ids is not None else None,
        exclude_team_ids=",".join(query.exclude_entity_ids) if query.exclude_entity_ids else None,
        api_key=query.api_key,
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    resolved: Final = _build_scope(
        TEAM_RESOLVER,
        query,
        team_scope.team_ids,
        team_scope.exclude_team_ids or (),
        team_scope.api_key_filter,
        team_scope.team_alias_metadata,
    )
    if query.tags is None and not query.exclude_tags and query.group_by != "tag":
        return resolved
    return ResolvedScope(
        scope=replace(
            resolved.scope,
            table=DailyActivityTable.TAG,
            entity_id_field="tag" if query.group_by == "tag" else "team_id",
            entity_ids=None,
            exclude_entity_ids=(),
            team_ids=resolved.scope.entity_ids,
            exclude_team_ids=resolved.scope.exclude_entity_ids,
            tags=query.tags,
            exclude_tags=query.exclude_tags,
        ),
        entity_metadata=None if query.group_by == "tag" else resolved.entity_metadata,
    )


async def _resolve_tag(
    user_api_key_dict: UserAPIKeyAuth, query: EntityQuery, prisma_client: PrismaClient
) -> EntityScopeResolution:
    if query.team_ids is not None or query.exclude_team_ids or query.group_by == "team":
        team_query: Final = replace(
            query,
            entity_ids=query.team_ids,
            exclude_entity_ids=query.exclude_team_ids,
            group_by=None,
        )
        resolved: Final = await _resolve_team(user_api_key_dict, team_query, prisma_client)
        if isinstance(resolved, ScopeDenied):
            return resolved
        return replace(
            resolved,
            scope=replace(
                resolved.scope,
                table=DailyActivityTable.TAG,
                entity_id_field="team_id" if query.group_by == "team" else "tag",
                entity_ids=None,
                exclude_entity_ids=(),
                team_ids=resolved.scope.entity_ids,
                exclude_team_ids=resolved.scope.exclude_entity_ids,
                tags=query.entity_ids,
                exclude_tags=query.exclude_entity_ids,
            ),
            entity_metadata=resolved.entity_metadata if query.group_by == "team" else None,
        )
    api_key_filter: Final = await get_tag_daily_activity_api_key_filter(
        prisma_client=prisma_client,
        user_api_key_dict=user_api_key_dict,
        requested_api_key=query.api_key,
    )
    return _build_scope(
        TAG_RESOLVER,
        query,
        query.entity_ids,
        query.exclude_entity_ids,
        api_key_filter,
        None,
    )


async def _resolve_organization(
    user_api_key_dict: UserAPIKeyAuth, query: EntityQuery, prisma_client: PrismaClient
) -> EntityScopeResolution:
    org_scope: Final = await resolve_organization_daily_activity_scope(
        organization_ids=query.entity_ids,
        prisma_client=prisma_client,
        user_api_key_dict=user_api_key_dict,
    )
    return _build_scope(
        ORGANIZATION_RESOLVER,
        query,
        org_scope.organization_ids,
        query.exclude_entity_ids,
        query.api_key,
        org_scope.organization_metadata,
    )


async def _resolve_customer(
    user_api_key_dict: UserAPIKeyAuth, query: EntityQuery, prisma_client: PrismaClient
) -> EntityScopeResolution:
    if not customer_daily_activity_is_admin(user_api_key_dict):
        return ScopeDenied(403, f"Admin-only endpoint. Your user role={user_api_key_dict.user_role}")
    customer_scope: Final = await resolve_customer_daily_activity_scope(
        end_user_ids=query.entity_ids,
        prisma_client=prisma_client,
    )
    return _build_scope(
        CUSTOMER_RESOLVER,
        query,
        customer_scope.end_user_ids,
        query.exclude_entity_ids,
        query.api_key,
        customer_scope.end_user_metadata,
    )


async def _resolve_agent(
    user_api_key_dict: UserAPIKeyAuth, query: EntityQuery, prisma_client: PrismaClient
) -> EntityScopeResolution:
    agent_scope: Final = await resolve_agent_daily_activity_scope(
        agent_ids=query.entity_ids,
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
    )
    return _build_scope(
        AGENT_RESOLVER,
        query,
        agent_scope.agent_ids,
        query.exclude_entity_ids,
        query.api_key,
        agent_scope.agent_metadata,
    )


def _user_query(
    start_date: str | None = Query(default=None, description="Start date in YYYY-MM-DD format"),
    end_date: str | None = Query(default=None, description="End date in YYYY-MM-DD format"),
    model: str | None = Query(default=None, description="Filter by specific model"),
    api_key: str | None = Query(default=None, description="Filter by specific API key"),
    user_id: str | None = Query(
        default=None,
        description="Filter by specific user ID. Admins can filter by any user or omit for global view. "
        "Non-admins must provide their own user_id.",
    ),
    timezone: int | None = Query(
        default=None,
        description="Timezone offset in minutes from UTC (e.g., 480 for PST). "
        "Matches JavaScript's Date.getTimezoneOffset() convention.",
    ),
    include_current_utc_day: bool = Query(
        default=False,
        description="When the range ends on the caller's current local day, extend it to "
        "today's UTC bucket so spend written after the caller's local midnight (in UTC "
        "terms) is included. Requires the timezone parameter. Historical ranges are "
        "never extended.",
    ),
) -> EntityQuery:
    return EntityQuery(
        entity_ids=(user_id,) if user_id is not None else None,
        exclude_entity_ids=(),
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        model=model,
        timezone_offset_minutes=timezone,
        include_current_utc_day=include_current_utc_day,
    )


def _team_query(
    team_ids: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    exclude_team_ids: str | None = None,
    timezone: int | None = None,
    tags: str | None = None,
    exclude_tags: str | None = None,
    group_by: Literal["team", "tag"] | None = None,
) -> EntityQuery:
    return EntityQuery(
        entity_ids=_query_ids(team_ids),
        exclude_entity_ids=_query_excluded_ids(exclude_team_ids),
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        model=model,
        timezone_offset_minutes=timezone,
        include_current_utc_day=False,
        tags=_query_ids(tags),
        exclude_tags=_query_excluded_ids(exclude_tags),
        group_by=group_by,
    )


def _tag_query(
    start_date: str | None = None,
    end_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    tags: str | None = None,
    timezone: int | None = None,
    team_ids: str | None = None,
    exclude_team_ids: str | None = None,
    exclude_tags: str | None = None,
    group_by: Literal["tag", "team"] | None = None,
) -> EntityQuery:
    return EntityQuery(
        entity_ids=_query_ids(tags),
        exclude_entity_ids=_query_excluded_ids(exclude_tags),
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        model=model,
        timezone_offset_minutes=timezone,
        include_current_utc_day=False,
        team_ids=_query_ids(team_ids),
        exclude_team_ids=_query_excluded_ids(exclude_team_ids),
        group_by=group_by,
    )


def _organization_query(
    organization_ids: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    exclude_organization_ids: str | None = None,
    timezone: int | None = None,
) -> EntityQuery:
    return EntityQuery(
        entity_ids=_query_ids(organization_ids),
        exclude_entity_ids=_query_excluded_ids(exclude_organization_ids),
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        model=model,
        timezone_offset_minutes=timezone,
        include_current_utc_day=False,
    )


def _customer_query(
    end_user_ids: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    exclude_end_user_ids: str | None = None,
    timezone: int | None = None,
) -> EntityQuery:
    return EntityQuery(
        entity_ids=_query_ids(end_user_ids),
        exclude_entity_ids=_query_excluded_ids(exclude_end_user_ids),
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        model=model,
        timezone_offset_minutes=timezone,
        include_current_utc_day=False,
    )


def _agent_query(
    agent_ids: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    exclude_agent_ids: str | None = None,
    timezone: int | None = None,
) -> EntityQuery:
    return EntityQuery(
        entity_ids=_query_ids(agent_ids),
        exclude_entity_ids=_query_excluded_ids(exclude_agent_ids),
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        model=model,
        timezone_offset_minutes=timezone,
        include_current_utc_day=False,
    )


USER_RESOLVER = EntityScopeResolver(
    entity="user",
    table=DailyActivityTable.USER,
    entity_id_field="user_id",
    route_prefixes=("/user",),
    tags=("Budget & Spend Tracking", "Internal User management"),
    query=_user_query,
    resolve=_resolve_user,
    include_entity_breakdown=False,
    operation_names=MappingProxyType(
        {
            "aggregated": "get_user_daily_activity_aggregated",
            "search": "get_user_daily_activity_aggregated_search",
            "key_page": "get_user_daily_activity_aggregated_keys",
            "model_top_keys": "get_user_daily_activity_model_top_keys",
            "export": "get_user_daily_activity_export",
            "cache_leakage_keys": "get_user_daily_activity_cache_leakage_keys",
        }
    ),
)
TEAM_RESOLVER = EntityScopeResolver(
    entity="team",
    table=DailyActivityTable.TEAM,
    entity_id_field="team_id",
    route_prefixes=("/team",),
    tags=("team management",),
    query=_team_query,
    resolve=_resolve_team,
    include_entity_breakdown=True,
    operation_names=MappingProxyType(
        {
            "aggregated": "get_team_daily_activity_aggregated",
            "search": "get_team_daily_activity_aggregated_search",
            "key_page": "get_team_daily_activity_aggregated_keys",
            "model_top_keys": "get_team_daily_activity_model_top_keys",
            "export": "get_team_daily_activity_export",
        }
    ),
)
TAG_RESOLVER = EntityScopeResolver(
    entity="tag",
    table=DailyActivityTable.TAG,
    entity_id_field="tag",
    route_prefixes=("/tag",),
    tags=("tag management",),
    query=_tag_query,
    resolve=_resolve_tag,
    include_entity_breakdown=True,
    operation_names=MappingProxyType(
        {
            "aggregated": "get_tag_daily_activity_aggregated",
            "search": "get_tag_daily_activity_aggregated_search",
            "key_page": "get_tag_daily_activity_aggregated_keys",
            "model_top_keys": "get_tag_daily_activity_model_top_keys",
            "export": "get_tag_daily_activity_export",
        }
    ),
)
ORGANIZATION_RESOLVER = EntityScopeResolver(
    entity="organization",
    table=DailyActivityTable.ORGANIZATION,
    entity_id_field="organization_id",
    route_prefixes=("/organization",),
    tags=("organization management",),
    query=_organization_query,
    resolve=_resolve_organization,
    include_entity_breakdown=True,
    operation_names=MappingProxyType(
        {
            "aggregated": "get_organization_daily_activity_aggregated",
            "search": "get_organization_daily_activity_aggregated_search",
            "key_page": "get_organization_daily_activity_aggregated_keys",
            "model_top_keys": "get_organization_daily_activity_model_top_keys",
            "export": "get_organization_daily_activity_export",
        }
    ),
)
CUSTOMER_RESOLVER = EntityScopeResolver(
    entity="customer",
    table=DailyActivityTable.CUSTOMER,
    entity_id_field="end_user_id",
    route_prefixes=("/customer", "/end_user"),
    tags=("Customer Management",),
    query=_customer_query,
    resolve=_resolve_customer,
    include_entity_breakdown=True,
    operation_names=MappingProxyType(
        {
            "aggregated": "get_customer_daily_activity_aggregated",
            "search": "get_customer_daily_activity_aggregated_search",
            "key_page": "get_customer_daily_activity_aggregated_keys",
            "model_top_keys": "get_customer_daily_activity_model_top_keys",
            "export": "get_customer_daily_activity_export",
        }
    ),
)
AGENT_RESOLVER = EntityScopeResolver(
    entity="agent",
    table=DailyActivityTable.AGENT,
    entity_id_field="agent_id",
    route_prefixes=("/agent",),
    tags=("Agent Management",),
    query=_agent_query,
    resolve=_resolve_agent,
    include_entity_breakdown=True,
    operation_names=MappingProxyType(
        {
            "aggregated": "get_agent_daily_activity_aggregated",
            "search": "get_agent_daily_activity_aggregated_search",
            "key_page": "get_agent_daily_activity_aggregated_keys",
            "model_top_keys": "get_agent_daily_activity_model_top_keys",
            "export": "get_agent_daily_activity_export",
        }
    ),
)
