import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Literal, cast

from fastapi import APIRouter, Depends, Query

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens
from litellm.proxy.spend_tracking.optimization import extract_cache_read_tokens
from litellm.proxy.spend_tracking.savings import compute_savings_spend
from litellm.types.proxy.management_endpoints.cost_optimization import (
    OptimizedRequestLog,
    OptimizedRequestLogsResponse,
)

router = APIRouter()


def _metadata_dict(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)  # cast-ok: JSON metadata is validated as a mapping above
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return (
                cast(Mapping[str, object], parsed)  # cast-ok: parsed JSON is checked as a dict
                if isinstance(parsed, dict)
                else {}
            )
        except (TypeError, ValueError):
            return {}
    return {}


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)


def build_optimized_request_log(row: Mapping[str, object]) -> OptimizedRequestLog | None:
    metadata = _metadata_dict(row.get("metadata"))
    cache_read_tokens = extract_cache_read_tokens(metadata)
    tokens_saved = extract_compression_saved_tokens(metadata)
    if cache_read_tokens <= 0 and tokens_saved <= 0:
        return None

    model = str(row.get("model", "") or "")
    provider_value = row.get("custom_llm_provider")
    provider = str(provider_value) if provider_value is not None else None
    savings_spend = compute_savings_spend(
        model=model,
        custom_llm_provider=provider,
        compression_saved_tokens=tokens_saved,
        cache_read_input_tokens=cache_read_tokens,
    )
    compression_savings = float(savings_spend.compression)
    caching_savings = float(savings_spend.prompt_caching)
    if tokens_saved > 0 and cache_read_tokens > 0:
        optimization_type: Literal["compression", "caching", "both"] = "both"
    elif tokens_saved > 0:
        optimization_type = "compression"
    else:
        optimization_type = "caching"

    spend_value = row.get("spend")
    spend = float(spend_value) if isinstance(spend_value, (int, float)) else 0.0
    timestamp = row.get("startTime")
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()
    else:
        timestamp = str(timestamp or "")

    return OptimizedRequestLog(
        request_id=str(row.get("request_id", "")),
        timestamp=timestamp,
        model=str(model or ""),
        total_tokens=_nonnegative_int(row.get("total_tokens")),
        optimization_type=optimization_type,
        spend=spend,
        savings=compression_savings + caching_savings,
        original_cost=spend + compression_savings + caching_savings,
        compression_savings_spend=compression_savings,
        prompt_caching_savings_spend=caching_savings,
        tokens_saved=tokens_saved,
        cache_read_tokens=cache_read_tokens,
    )


def _cache_read_filter_sql() -> str:
    candidates = (
        "(metadata->'usage_object'->>'cache_read_input_tokens')",
        "(metadata->'usage_object'->'prompt_tokens_details'->>'cached_tokens')",
        "(metadata->'additional_usage_values'->>'cache_read_input_tokens')",
        "(metadata->'additional_usage_values'->'prompt_tokens_details'->>'cached_tokens')",
    )
    return " OR ".join(f"({candidate}) ~ '^[0-9]+$' AND ({candidate})::bigint > 0" for candidate in candidates)


def _optimized_where_sql() -> str:
    return f"""(
        (metadata->'compression_savings' IS NOT NULL
         AND metadata->'compression_savings' <> 'null'::jsonb)
        OR {_cache_read_filter_sql()}
    )"""


def _parse_date(value: str | None, *, end: bool = False) -> datetime | None:
    if not value:
        return None
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S" if " " in value else "%Y-%m-%d")
    parsed = parsed.replace(tzinfo=timezone.utc)
    if end and " " not in value:
        parsed += timedelta(days=1)
    return parsed


@router.get(
    "/cost_optimization/usage/logs",
    tags=["Cost Optimization"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=OptimizedRequestLogsResponse,
)
async def cost_optimization_usage_logs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
) -> OptimizedRequestLogsResponse:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return OptimizedRequestLogsResponse(logs=[], total=0, page=page, page_size=page_size, total_pages=0)

    conditions = [_optimized_where_sql()]
    params: list[object] = []
    if start_date:
        conditions.append('"startTime" >= $%d' % (len(params) + 1))
        params.append(_parse_date(start_date))
    if end_date:
        conditions.append('"startTime" < $%d' % (len(params) + 1))
        params.append(_parse_date(end_date, end=True))
    where_sql = " AND ".join(conditions)

    query_raw: Callable[..., Awaitable[object]] = cast(  # cast-ok: Prisma exposes query_raw dynamically
        Callable[..., Awaitable[object]], prisma_client.db.query_raw
    )
    total_rows = cast(  # cast-ok: Prisma raw query returns rows with the selected columns
        list[Mapping[str, object]],
        await query_raw(
            f'SELECT COUNT(*)::bigint AS total FROM "LiteLLM_SpendLogs" WHERE {where_sql}',
            *params,
        ),
    )
    total_value = total_rows[0].get("total") if total_rows else 0
    total = int(total_value) if isinstance(total_value, (int, float)) else 0
    offset_placeholder = len(params) + 1
    limit_placeholder = len(params) + 2
    rows = cast(  # cast-ok: Prisma raw query returns rows with the selected columns
        list[Mapping[str, object]],
        await query_raw(
            f"""SELECT request_id, "startTime", model, custom_llm_provider, total_tokens, spend, metadata
            FROM "LiteLLM_SpendLogs"
            WHERE {where_sql}
            ORDER BY "startTime" DESC
            LIMIT ${limit_placeholder} OFFSET ${offset_placeholder}""",
            *params,
            page_size,
            (page - 1) * page_size,
        ),
    )
    logs = [entry for row in rows if (entry := build_optimized_request_log(row)) is not None]
    return OptimizedRequestLogsResponse(
        logs=logs,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )
