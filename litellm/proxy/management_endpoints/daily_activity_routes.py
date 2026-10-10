import csv
import io
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, fields, replace
from datetime import date, datetime
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from litellm import constants
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_daily_activity import (
    InvalidDateRange,
    ScopeDenied,
    daily_activity_repository,
    get_daily_activity_aggregated,
    parse_canonical_date_range,
    raise_public,
    spend_logs_window,
)
from litellm.proxy.management_endpoints.daily_activity_scopes import (
    AGENT_RESOLVER,
    CUSTOMER_RESOLVER,
    ORGANIZATION_RESOLVER,
    TAG_RESOLVER,
    TEAM_RESOLVER,
    USER_RESOLVER,
    EntityQuery,
    EntityScopeResolver,
    ResolvedScope,
)
from litellm.proxy.management_endpoints.team_endpoints import aggregated_date_range_error
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.proxy.utils import PrismaClient, get_prisma_client_or_throw
from litellm.repositories.daily_activity_repository import DailyActivityRepository
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    CacheLeakageKeysResponse,
    DailyActivityKeyPageResponse,
    DailyActivityKeySearchResponse,
    DailyActivityUserPageResponse,
    KeyActivityRow,
    KeyMetadata,
    KeySpendActivityRow,
    KeySpendMetrics,
    ModelTopKeysResponse,
    SpendAnalyticsPaginatedResponse,
    SpendMetrics,
    UserActivityRow,
)
from litellm.types.repositories.daily_activity import (
    ExportRow,
    ExportType,
    KeyMetadataRow,
    KeySpendRow,
    UserMetadataRow,
    UserSpendRow,
)

router = APIRouter()


def get_daily_activity_prisma_client() -> PrismaClient:
    return get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)


def get_daily_activity_repository() -> DailyActivityRepository:
    return daily_activity_repository(get_daily_activity_prisma_client())


def _date_range_error(query: EntityQuery, *, user_aggregated: bool) -> InvalidDateRange | None:
    if user_aggregated:
        date_range: Final = parse_canonical_date_range(query.start_date, query.end_date)
        return date_range if isinstance(date_range, InvalidDateRange) else None

    range_error: Final[str | None] = aggregated_date_range_error(query.start_date, query.end_date)
    return None if range_error is None else InvalidDateRange(reason=range_error)


async def _resolved_scope(
    resolver: EntityScopeResolver,
    query: EntityQuery,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    *,
    user_aggregated: bool,
) -> ResolvedScope:
    date_error: Final[InvalidDateRange | None] = _date_range_error(query, user_aggregated=user_aggregated)
    if date_error is not None:
        raise_public(date_error)
    result: ResolvedScope | ScopeDenied = await resolver.resolve(user_api_key_dict, query, prisma_client)
    if isinstance(result, ScopeDenied):
        raise_public(result)
    return result


def _sum_metrics(metrics: Sequence[SpendMetrics]) -> SpendMetrics:
    return SpendMetrics(
        spend=sum(metric.spend for metric in metrics),
        flat_cost=sum(metric.flat_cost for metric in metrics),
        prompt_tokens=sum(metric.prompt_tokens for metric in metrics),
        completion_tokens=sum(metric.completion_tokens for metric in metrics),
        cache_read_input_tokens=sum(metric.cache_read_input_tokens for metric in metrics),
        cache_creation_input_tokens=sum(metric.cache_creation_input_tokens for metric in metrics),
        compression_saved_tokens=sum(metric.compression_saved_tokens for metric in metrics),
        compression_savings_spend=sum(metric.compression_savings_spend for metric in metrics),
        prompt_caching_savings_spend=sum(metric.prompt_caching_savings_spend for metric in metrics),
        gateway_injected_caching_savings_spend=sum(metric.gateway_injected_caching_savings_spend for metric in metrics),
        autorouter_savings_spend=sum(metric.autorouter_savings_spend for metric in metrics),
        total_tokens=sum(metric.total_tokens for metric in metrics),
        successful_requests=sum(metric.successful_requests for metric in metrics),
        failed_requests=sum(metric.failed_requests for metric in metrics),
        api_requests=sum(metric.api_requests for metric in metrics),
        total_response_time_ms=sum(metric.total_response_time_ms for metric in metrics),
        timed_requests=sum(metric.timed_requests for metric in metrics),
    )


def _key_metadata(api_key: str, metadata: Mapping[str, KeyMetadataRow]) -> KeyMetadata:
    row: Final[KeyMetadataRow | None] = metadata.get(api_key)
    if row is None:
        return KeyMetadata()
    return KeyMetadata(
        key_alias=row.key_alias,
        team_id=row.team_id,
        user_id=row.user_id,
        user_email=row.user_email,
        key_exists=row.key_exists,
    )


def _key_activity_row(row: KeySpendRow, metadata: Mapping[str, KeyMetadataRow]) -> KeySpendActivityRow:
    return KeySpendActivityRow(
        api_key=row.api_key,
        metrics=KeySpendMetrics(
            spend=row.spend,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            total_tokens=row.total_tokens,
            api_requests=row.api_requests,
            successful_requests=row.successful_requests,
            failed_requests=row.failed_requests,
            cache_read_input_tokens=row.cache_read_input_tokens,
            cache_creation_input_tokens=row.cache_creation_input_tokens,
        ),
        metadata=_key_metadata(row.api_key, metadata),
    )


async def _key_activity_rows(
    repository: DailyActivityRepository,
    rows: Sequence[KeySpendRow],
    resolved_scope: ResolvedScope,
) -> list[KeySpendActivityRow]:
    spend_window: Final[tuple[datetime, datetime] | None] = spend_logs_window(
        frozenset((resolved_scope.scope.start_date, resolved_scope.scope.end_date))
    )
    metadata: Final[Mapping[str, KeyMetadataRow]] = await repository.key_metadata(
        frozenset(row.api_key for row in rows),
        spend_window,
    )
    return [_key_activity_row(row, metadata) for row in rows]


def _user_activity_row(row: UserSpendRow, metadata: Mapping[str, UserMetadataRow]) -> UserActivityRow:
    meta: Final = metadata.get(row.user_id) if row.user_id is not None else None
    return UserActivityRow(
        user_id=row.user_id,
        user_email=meta.user_email if meta is not None else None,
        user_alias=meta.user_alias if meta is not None else None,
        spend=row.spend,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.total_tokens,
        api_requests=row.api_requests,
        successful_requests=row.successful_requests,
        failed_requests=row.failed_requests,
    )


async def _user_activity_rows(
    repository: DailyActivityRepository,
    rows: Sequence[UserSpendRow],
) -> Sequence[UserActivityRow]:
    metadata: Final[Mapping[str, UserMetadataRow]] = await repository.user_metadata(
        frozenset(user_id for row in rows if (user_id := row.user_id) is not None)
    )
    return [_user_activity_row(row, metadata) for row in rows]


def _export_filename(
    entity: str,
    start_date: date,
    end_date: date,
    export_type: ExportType,
    file_format: Literal["csv", "json"],
) -> str:
    extension: Final[str] = "csv" if file_format == "csv" else "json"
    return f"{entity}-usage-{start_date.isoformat()}-{end_date.isoformat()}-{export_type.value}.{extension}"


def _content_disposition(
    entity: str,
    start_date: date,
    end_date: date,
    export_type: ExportType,
    file_format: Literal["csv", "json"],
) -> str:
    filename: Final = _export_filename(entity, start_date, end_date, export_type, file_format)
    return f'attachment; filename="{filename}"'


def _fold_key_metrics(api_key: str, response: SpendAnalyticsPaginatedResponse) -> KeyActivityRow | None:
    metrics: Final[tuple[SpendMetrics, ...]] = tuple(
        day.breakdown.api_keys[api_key].metrics for day in response.results if api_key in day.breakdown.api_keys
    )
    if not metrics:
        return None
    metadata: Final[KeyMetadata] = next(
        day.breakdown.api_keys[api_key].metadata for day in response.results if api_key in day.breakdown.api_keys
    )
    return KeyActivityRow(api_key=api_key, metrics=_sum_metrics(metrics), metadata=metadata)


def _search_rows(keys: Sequence[str], response: SpendAnalyticsPaginatedResponse) -> list[KeyActivityRow]:
    return [row for key in keys if (row := _fold_key_metrics(key, response)) is not None]


def _csv_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


def _csv_row(values: Sequence[object]) -> bytes:
    output: Final[io.StringIO] = io.StringIO(newline="")
    csv.writer(output, lineterminator="\r\n").writerow(tuple(_csv_cell(value) for value in values))
    return output.getvalue().encode()


def _stream_export_rows(
    first_row: ExportRow | None,
    rows: AsyncIterator[ExportRow],
    file_format: Literal["csv", "json"],
) -> AsyncIterator[bytes]:
    async def stream() -> AsyncIterator[bytes]:
        if file_format == "csv":
            yield _csv_row(tuple(field.name for field in fields(ExportRow)))
            if first_row is not None:
                yield _csv_row(tuple(asdict(first_row).values()))
            async for row in rows:
                yield _csv_row(tuple(asdict(row).values()))
            return

        if first_row is None:
            yield b"[]"
            return
        yield b"[" + json.dumps(jsonable_encoder(first_row), separators=(",", ":")).encode()
        async for row in rows:
            yield b"," + json.dumps(jsonable_encoder(row), separators=(",", ":")).encode()
        yield b"]"

    return stream()


def _register_aggregated_route(router: APIRouter, resolver: EntityScopeResolver, prefix: str) -> None:
    @management_endpoint_wrapper
    async def aggregated(
        entity_query: Annotated[EntityQuery, Depends(resolver.query)],
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
        repository: Annotated[DailyActivityRepository, Depends(get_daily_activity_repository)],
        prisma_client: Annotated[PrismaClient, Depends(get_daily_activity_prisma_client)],
        api_key_limit: Annotated[
            int, Query(ge=1, le=constants.USAGE_TOP_API_KEYS_MAX)
        ] = constants.USAGE_TOP_API_KEYS_DEFAULT,
    ) -> SpendAnalyticsPaginatedResponse:
        try:
            resolved: ResolvedScope = await _resolved_scope(
                resolver,
                entity_query,
                user_api_key_dict,
                prisma_client,
                user_aggregated=resolver.entity == "user",
            )
            return await get_daily_activity_aggregated(
                repository,
                resolved.scope,
                entity_metadata_field=resolved.entity_metadata,
                include_entity_breakdown=resolver.include_entity_breakdown,
                api_key_limit=api_key_limit,
            )
        except HTTPException:
            raise
        except Exception as exc:
            verbose_proxy_logger.exception("Daily activity aggregation failed: %s", exc)
            raise HTTPException(status_code=500, detail={"error": f"Failed to fetch analytics: {exc}"})

    router.add_api_route(
        f"{prefix}/daily/activity/aggregated",
        aggregated,
        methods=["GET"],
        name=resolver.operation_names["aggregated"],
        tags=list(resolver.tags),
        dependencies=(Depends(user_api_key_auth),),
        response_model=SpendAnalyticsPaginatedResponse,
        include_in_schema=prefix != "/end_user",
    )

    if resolver.entity == "user":
        aggregated.__doc__ = (
            "Aggregated analytics for a user's daily activity without pagination.\n"
            "Returns the same response shape as the paginated endpoint with page metadata set to single-page.\n\n"
            "Reads daily spend records that only ever accumulate and are never affected by budget\n"
            "resets. Their total can legitimately exceed the `spend` field returned by\n"
            "`/v2/user/info`, which is a running budget counter that every budget reset sets back\n"
            "to zero (or to the overage above `max_budget` when `budget_rollover` is enabled)."
        )
    elif resolver.entity == "team":
        aggregated.__doc__ = (
            "Aggregated daily activity for teams without pagination, including per-team breakdown.\n\n"
            "One SQL GROUPING SETS pass returns every day in the range regardless of row\n"
            "volume, so callers never reassemble pages. Same response shape as the\n"
            "paginated endpoint with page metadata pinned to a single page.\n\n"
            "Args:\n"
            "    team_ids (Optional[str]): Comma-separated list of team IDs to filter by. If not provided, "
            "returns data for all teams.\n"
            "    start_date (Optional[str]): Start date for the activity period (YYYY-MM-DD).\n"
            "    end_date (Optional[str]): End date for the activity period (YYYY-MM-DD).\n"
            "    model (Optional[str]): Filter by model name.\n"
            "    api_key (Optional[str]): Filter by API key.\n"
            "    exclude_team_ids (Optional[str]): Comma-separated list of team IDs to exclude.\n"
            "    timezone (Optional[int]): Timezone offset in minutes from UTC, matching JavaScript's "
            "Date.getTimezoneOffset() convention.\n"
            "Returns:\n"
            "    SpendAnalyticsPaginatedResponse: Response containing all daily activity data for the range."
        )


def _register_key_page_route(router: APIRouter, resolver: EntityScopeResolver, prefix: str) -> None:
    @management_endpoint_wrapper
    async def key_page(
        entity_query: Annotated[EntityQuery, Depends(resolver.query)],
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
        repository: Annotated[DailyActivityRepository, Depends(get_daily_activity_repository)],
        prisma_client: Annotated[PrismaClient, Depends(get_daily_activity_prisma_client)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=constants.USAGE_KEY_PAGE_MAX)] = constants.USAGE_KEY_PAGE_DEFAULT,
    ) -> DailyActivityKeyPageResponse:
        try:
            resolved: Final = await _resolved_scope(
                resolver,
                entity_query,
                user_api_key_dict,
                prisma_client,
                user_aggregated=False,
            )
            page: Final = await repository.key_page(resolved.scope, offset=offset, limit=limit)
            return DailyActivityKeyPageResponse(
                api_keys=await _key_activity_rows(repository, page.rows, resolved),
                total_api_keys=page.total_api_keys,
                offset=offset,
                limit=limit,
            )
        except HTTPException:
            raise
        except Exception as exc:
            verbose_proxy_logger.exception("Daily activity key page failed: %s", exc)
            raise HTTPException(status_code=500, detail={"error": f"Failed to fetch analytics: {exc}"})

    router.add_api_route(
        f"{prefix}/daily/activity/aggregated/keys",
        key_page,
        methods=["GET"],
        name=resolver.operation_names["key_page"],
        tags=list(resolver.tags),
        dependencies=(Depends(user_api_key_auth),),
        response_model=DailyActivityKeyPageResponse,
        include_in_schema=prefix != "/end_user",
    )


def _register_search_route(router: APIRouter, resolver: EntityScopeResolver, prefix: str) -> None:
    @management_endpoint_wrapper
    async def search(
        entity_query: Annotated[EntityQuery, Depends(resolver.query)],
        search: Annotated[str, Query(min_length=1)],
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
        repository: Annotated[DailyActivityRepository, Depends(get_daily_activity_repository)],
        prisma_client: Annotated[PrismaClient, Depends(get_daily_activity_prisma_client)],
        limit: Annotated[int, Query(ge=1, le=constants.USAGE_KEY_SEARCH_MAX)] = constants.USAGE_KEY_SEARCH_DEFAULT,
    ) -> DailyActivityKeySearchResponse:
        try:
            resolved: ResolvedScope = await _resolved_scope(
                resolver,
                entity_query,
                user_api_key_dict,
                prisma_client,
                user_aggregated=False,
            )
            keys: tuple[str, ...] = await repository.search_keys(
                resolved.scope,
                search=search,
                limit=limit,
            )
            if not keys:
                return DailyActivityKeySearchResponse(api_keys=[])
            search_response: SpendAnalyticsPaginatedResponse = await get_daily_activity_aggregated(
                repository,
                replace(resolved.scope, api_keys=keys),
                include_entity_breakdown=False,
            )
            return DailyActivityKeySearchResponse(api_keys=_search_rows(keys, search_response))
        except HTTPException:
            raise
        except Exception as exc:
            verbose_proxy_logger.exception("Daily activity key search failed: %s", exc)
            raise HTTPException(status_code=500, detail={"error": f"Failed to fetch analytics: {exc}"})

    router.add_api_route(
        f"{prefix}/daily/activity/aggregated/search",
        search,
        methods=["GET"],
        name=resolver.operation_names["search"],
        tags=list(resolver.tags),
        dependencies=(Depends(user_api_key_auth),),
        response_model=DailyActivityKeySearchResponse,
        include_in_schema=prefix != "/end_user",
    )


def _register_model_top_keys_route(router: APIRouter, resolver: EntityScopeResolver, prefix: str) -> None:
    @management_endpoint_wrapper
    async def model_top_keys(
        entity_query: Annotated[EntityQuery, Depends(resolver.query)],
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
        repository: Annotated[DailyActivityRepository, Depends(get_daily_activity_repository)],
        prisma_client: Annotated[PrismaClient, Depends(get_daily_activity_prisma_client)],
        model_group: Annotated[str, Query(min_length=1)],
        by_model_group: Annotated[bool, Query()] = True,
        limit: Annotated[int, Query(ge=1, le=constants.USAGE_MODEL_TOP_KEYS_MAX)] = (
            constants.USAGE_MODEL_TOP_KEYS_DEFAULT
        ),
    ) -> ModelTopKeysResponse:
        try:
            resolved: ResolvedScope = await _resolved_scope(
                resolver,
                entity_query,
                user_api_key_dict,
                prisma_client,
                user_aggregated=False,
            )
            rows: tuple[KeySpendRow, ...] = await repository.model_top_keys(
                resolved.scope,
                model_group=model_group,
                by_model_group=by_model_group,
                limit=limit,
            )
            return ModelTopKeysResponse(
                model=model_group,
                by_model_group=by_model_group,
                api_keys=await _key_activity_rows(repository, rows, resolved),
            )
        except HTTPException:
            raise
        except Exception as exc:
            verbose_proxy_logger.exception("Daily activity model top keys failed: %s", exc)
            raise HTTPException(status_code=500, detail={"error": f"Failed to fetch analytics: {exc}"})

    router.add_api_route(
        f"{prefix}/daily/activity/aggregated/model_top_keys",
        model_top_keys,
        methods=["GET"],
        name=resolver.operation_names["model_top_keys"],
        tags=list(resolver.tags),
        dependencies=(Depends(user_api_key_auth),),
        response_model=ModelTopKeysResponse,
        include_in_schema=prefix != "/end_user",
    )


def _register_export_route(router: APIRouter, resolver: EntityScopeResolver, prefix: str) -> None:
    @management_endpoint_wrapper
    async def export(
        entity_query: Annotated[EntityQuery, Depends(resolver.query)],
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
        repository: Annotated[DailyActivityRepository, Depends(get_daily_activity_repository)],
        prisma_client: Annotated[PrismaClient, Depends(get_daily_activity_prisma_client)],
        export_type: Annotated[ExportType, Query()] = ExportType.DAILY,
        file_format: Annotated[Literal["csv", "json"], Query(alias="format")] = "csv",
    ) -> StreamingResponse:
        try:
            resolved: ResolvedScope = await _resolved_scope(
                resolver,
                entity_query,
                user_api_key_dict,
                prisma_client,
                user_aggregated=False,
            )
            rows: Final = repository.export_rows(resolved.scope, export_type=export_type)
            first_row: Final = await anext(rows, None)
            return StreamingResponse(
                _stream_export_rows(first_row, rows, file_format),
                media_type="text/csv" if file_format == "csv" else "application/json",
                headers={
                    "Cache-Control": "no-store",
                    "Content-Disposition": _content_disposition(
                        resolver.entity,
                        date.fromisoformat(resolved.scope.start_date),
                        date.fromisoformat(resolved.scope.end_date),
                        export_type,
                        file_format,
                    ),
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            verbose_proxy_logger.exception("Daily activity export failed: %s", exc)
            raise HTTPException(status_code=500, detail={"error": f"Failed to fetch analytics: {exc}"})

    router.add_api_route(
        f"{prefix}/daily/activity/export",
        export,
        methods=["GET"],
        name=resolver.operation_names["export"],
        tags=list(resolver.tags),
        dependencies=(Depends(user_api_key_auth),),
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "Streamed daily activity export",
                "content": {
                    "text/csv": {"schema": {"type": "string"}},
                    "application/json": {"schema": {"type": "array", "items": {"type": "object"}}},
                },
            }
        },
        include_in_schema=prefix != "/end_user",
    )


def _register_user_page_route(router: APIRouter, resolver: EntityScopeResolver, prefix: str) -> None:
    @management_endpoint_wrapper
    async def user_page(
        entity_query: Annotated[EntityQuery, Depends(resolver.query)],
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
        repository: Annotated[DailyActivityRepository, Depends(get_daily_activity_repository)],
        prisma_client: Annotated[PrismaClient, Depends(get_daily_activity_prisma_client)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=constants.USAGE_USER_PAGE_MAX)] = constants.USAGE_USER_PAGE_DEFAULT,
    ) -> DailyActivityUserPageResponse:
        try:
            resolved: Final = await _resolved_scope(
                resolver,
                entity_query,
                user_api_key_dict,
                prisma_client,
                user_aggregated=False,
            )
            page: Final = await repository.user_page(resolved.scope, offset=offset, limit=limit)
            return DailyActivityUserPageResponse(
                users=await _user_activity_rows(repository, page.rows),
                total_users=page.total_users,
                offset=offset,
                limit=limit,
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001, LIT003  # mirrors the sibling daily-activity routes' 500 mapping
            verbose_proxy_logger.exception("Daily activity user page failed: %s", exc)
            raise HTTPException(status_code=500, detail={"error": f"Failed to fetch analytics: {exc}"})

    router.add_api_route(
        f"{prefix}/daily/activity/aggregated/users",
        user_page,
        methods=["GET"],
        name=resolver.operation_names["user_page"],
        tags=list(resolver.tags),
        dependencies=(Depends(user_api_key_auth),),
        response_model=DailyActivityUserPageResponse,
        include_in_schema=prefix != "/end_user",
    )


def _register_cache_leakage_route(router: APIRouter, resolver: EntityScopeResolver, prefix: str) -> None:
    @management_endpoint_wrapper
    async def cache_leakage_keys(
        entity_query: Annotated[EntityQuery, Depends(resolver.query)],
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
        repository: Annotated[DailyActivityRepository, Depends(get_daily_activity_repository)],
        prisma_client: Annotated[PrismaClient, Depends(get_daily_activity_prisma_client)],
        limit: Annotated[
            int, Query(ge=1, le=constants.USAGE_CACHE_LEAKAGE_KEYS_MAX)
        ] = constants.USAGE_CACHE_LEAKAGE_KEYS_DEFAULT,
    ) -> CacheLeakageKeysResponse:
        try:
            resolved: ResolvedScope = await _resolved_scope(
                resolver,
                entity_query,
                user_api_key_dict,
                prisma_client,
                user_aggregated=False,
            )
            rows: tuple[KeySpendRow, ...] = await repository.cache_leakage_keys(
                resolved.scope,
                limit=limit,
            )
            return CacheLeakageKeysResponse(
                api_keys=await _key_activity_rows(repository, rows, resolved),
            )
        except HTTPException:
            raise
        except Exception as exc:
            verbose_proxy_logger.exception("Daily activity cache leakage keys failed: %s", exc)
            raise HTTPException(status_code=500, detail={"error": f"Failed to fetch analytics: {exc}"})

    router.add_api_route(
        f"{prefix}/daily/activity/aggregated/cache_leakage_keys",
        cache_leakage_keys,
        methods=["GET"],
        name=resolver.operation_names["cache_leakage_keys"],
        tags=list(resolver.tags),
        dependencies=(Depends(user_api_key_auth),),
        response_model=CacheLeakageKeysResponse,
        include_in_schema=prefix != "/end_user",
    )


def register_daily_activity_routes(router: APIRouter, resolver: EntityScopeResolver) -> None:
    for prefix in resolver.route_prefixes:
        _register_aggregated_route(router, resolver, prefix)
        _register_key_page_route(router, resolver, prefix)
        _register_search_route(router, resolver, prefix)
        _register_model_top_keys_route(router, resolver, prefix)
        _register_export_route(router, resolver, prefix)
        if resolver.entity == "user":
            _register_user_page_route(router, resolver, prefix)
            _register_cache_leakage_route(router, resolver, prefix)


for _resolver in (
    USER_RESOLVER,
    TEAM_RESOLVER,
    TAG_RESOLVER,
    ORGANIZATION_RESOLVER,
    CUSTOMER_RESOLVER,
    AGENT_RESOLVER,
):
    register_daily_activity_routes(router, _resolver)
