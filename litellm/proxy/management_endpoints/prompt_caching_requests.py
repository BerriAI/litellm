from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Json, TypeAdapter

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.spend_tracking.savings import (
    extract_cache_creation_tokens,
    extract_cache_read_tokens,
    marks_gateway_injection,
    prompt_caching_savings_for_request,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import (
    _query_raw_rows,  # pyright: ignore[reportPrivateUsage]  # existing typed spend-query adapter; rows validated below
)
from litellm.types.integrations.anthropic_cache_control_hook import GATEWAY_INJECTED_CACHE_METADATA_KEY
from litellm.types.management_endpoints.prompt_caching_requests import (
    PromptCachingRequest,
    PromptCachingRequestCursor,
    PromptCachingRequestFilter,
    PromptCachingRequestsResponse,
)

if TYPE_CHECKING:
    from litellm.router import Router

router: Final = APIRouter()


def _numeric_token_sql(path: str) -> str:
    value: Final = f"metadata #> '{{usage_object,{path}}}'"
    return (
        f"CASE WHEN jsonb_typeof({value}) = 'number' THEN ({value} #>> '{{}}')::numeric "
        f"WHEN {value} = 'true'::jsonb THEN 1 WHEN {value} = 'false'::jsonb THEN 0 END"
    )


def _cache_tokens_sql(*paths: str) -> str:
    candidates: Final = ", ".join(f"NULLIF(({_numeric_token_sql(path)}), 0)" for path in paths)
    return f"TRUNC(COALESCE({candidates}, 0))"


_CACHE_READ_SQL: Final = _cache_tokens_sql("cache_read_input_tokens", "prompt_tokens_details,cached_tokens")
_CACHE_CREATION_SQL: Final = _cache_tokens_sql(
    "cache_creation_input_tokens",
    "prompt_tokens_details,cache_write_tokens",
    "prompt_tokens_details,cache_creation_tokens",
)
_GATEWAY_INJECTED_SQL: Final = (
    f"(jsonb_typeof(metadata->'{GATEWAY_INJECTED_CACHE_METADATA_KEY}') = 'string' "
    f"AND (metadata->>'{GATEWAY_INJECTED_CACHE_METADATA_KEY}' = '' "
    f"OR metadata->>'{GATEWAY_INJECTED_CACHE_METADATA_KEY}' = model_id))"
)
_FILTER_SQL: Final = MappingProxyType(
    {
        "all": f"({_GATEWAY_INJECTED_SQL} OR {_CACHE_READ_SQL} > 0 OR {_CACHE_CREATION_SQL} > 0)",
        "injected": _GATEWAY_INJECTED_SQL,
        "hits": f"{_CACHE_READ_SQL} > 0",
    }
)


def prompt_caching_requests_sql(filter: PromptCachingRequestFilter) -> str:
    return f"""
        SELECT request_id, "startTime" AS start_time, "endTime" AS end_time,
               model, model_id, custom_llm_provider, spend,
               CASE WHEN jsonb_typeof(metadata->'usage_object') = 'object'
                    THEN metadata->'usage_object' END AS usage_object,
               CASE WHEN jsonb_typeof(metadata->'cost_breakdown') = 'object'
                    THEN metadata->'cost_breakdown' END AS cost_breakdown,
               CASE WHEN jsonb_typeof(metadata->'{GATEWAY_INJECTED_CACHE_METADATA_KEY}') = 'string'
                    THEN metadata->>'{GATEWAY_INJECTED_CACHE_METADATA_KEY}' END AS gateway_marker
        FROM "LiteLLM_SpendLogs"
        WHERE "startTime" >= ($1::text::timestamptz AT TIME ZONE 'UTC')
          AND "startTime" <= ($2::text::timestamptz AT TIME ZONE 'UTC')
          AND COALESCE(LOWER(cache_hit), 'false') != 'true'
          AND {_FILTER_SQL[filter]}
          AND ($4::text::timestamptz IS NULL OR
               ("startTime", request_id) < (($4::text::timestamptz AT TIME ZONE 'UTC'), $5::text))
        ORDER BY "startTime" DESC, request_id DESC
        LIMIT $3::integer
    """


class _PromptCachingRow(BaseModel):
    request_id: str
    start_time: datetime
    end_time: datetime
    model: str
    model_id: str | None
    custom_llm_provider: str | None
    spend: float
    usage_object: Json[Mapping[str, object]] | Mapping[str, object] | None
    cost_breakdown: Json[Mapping[str, object]] | Mapping[str, object] | None
    gateway_marker: str | None


_REQUEST_ROWS: Final = TypeAdapter(tuple[_PromptCachingRow, ...])


def _request_result(row: _PromptCachingRow, llm_router: "Callable[[], Router | None]") -> PromptCachingRequest:
    return PromptCachingRequest(
        request_id=row.request_id,
        start_time=row.start_time.replace(tzinfo=timezone.utc) if row.start_time.tzinfo is None else row.start_time,
        model=row.model,
        gateway_injected=marks_gateway_injection(
            MappingProxyType({GATEWAY_INJECTED_CACHE_METADATA_KEY: row.gateway_marker}), row.model_id
        ),
        cache_read_tokens=extract_cache_read_tokens(row.usage_object),
        cache_creation_tokens=extract_cache_creation_tokens(row.usage_object),
        spend=row.spend,
        net_savings=prompt_caching_savings_for_request(
            model=row.model,
            custom_llm_provider=row.custom_llm_provider,
            usage_object=row.usage_object,
            model_id=row.model_id,
            llm_router=llm_router,
            cost_breakdown=row.cost_breakdown,
            billed_at=row.end_time,
        ),
    )


@router.get(
    "/cost_optimization/prompt_caching/requests",
    tags=["Cost Optimization"],  # mutable-ok: FastAPI's route API requires a list
    response_model=PromptCachingRequestsResponse,
)
async def get_prompt_caching_requests(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: datetime,
    end_date: datetime,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
    filter: PromptCachingRequestFilter = "all",
    cursor_start_time: datetime | None = None,
    cursor_request_id: Annotated[str | None, Query(min_length=1)] = None,
) -> PromptCachingRequestsResponse:
    from litellm.proxy.proxy_server import llm_router, prisma_client

    if not user_api_key_has_admin_view(user_api_key_dict):
        raise HTTPException(status_code=403, detail="Only proxy admin roles can view prompt caching requests")
    if (cursor_start_time is None) != (cursor_request_id is None):
        raise HTTPException(status_code=400, detail="cursor_start_time and cursor_request_id must be provided together")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    start: Final = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date
    end: Final = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date
    if end < start:
        raise HTTPException(status_code=400, detail="end_date must not be earlier than start_date")
    cursor_time: Final = (
        cursor_start_time.replace(tzinfo=timezone.utc)
        if cursor_start_time is not None and cursor_start_time.tzinfo is None
        else cursor_start_time
    )
    rows: Final = _REQUEST_ROWS.validate_python(
        await _query_raw_rows(
            prisma_client,
            prompt_caching_requests_sql(filter),
            start.isoformat(),
            end.isoformat(),
            page_size + 1,
            cursor_time.isoformat() if cursor_time is not None else None,
            cursor_request_id,
        )
        or ()
    )

    def current_router() -> "Router | None":
        return llm_router

    requests: Final = tuple(_request_result(row, current_router) for row in rows[:page_size])
    has_more: Final = len(rows) > page_size
    return PromptCachingRequestsResponse(
        requests=requests,
        page_size=page_size,
        has_more=has_more,
        next_cursor=PromptCachingRequestCursor(start_time=requests[-1].start_time, request_id=requests[-1].request_id)
        if has_more
        else None,
    )
