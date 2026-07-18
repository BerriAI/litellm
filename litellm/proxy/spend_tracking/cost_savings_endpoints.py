"""
Cost savings analytics for prompt caching and prompt compression.

Dollarizes optimization token counts recorded in the daily spend aggregates
(cache_read_input_tokens, cache_creation_input_tokens, compression_saved_tokens)
using the model cost map:

- caching savings (net) = cache_read * (input_price - cache_read_price)
                          - cache_creation * (cache_creation_price - input_price)
- compression savings   = compression_saved * input_price
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import groupby
from typing import Annotated

import fastapi
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_utils import (
    _user_has_admin_view,
    require_caller_user_id_for_non_admin,
)
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.proxy.spend_tracking.cache_savings import (
    extract_cache_creation_tokens,
    extract_cache_read_tokens,
)
from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens
from litellm.types.proxy.cost_savings_endpoints import (
    CostSavingsActivityResponse,
    CostSavingsMetrics,
    DailyCostSavings,
    OptimizationType,
    OptimizedRequestSummary,
    RecentOptimizedRequestsResponse,
)

router = APIRouter(tags=["Budget & Spend Tracking"])

RECENT_REQUESTS_SCAN_WINDOW = 500

_ACTIVITY_SQL = (
    "SELECT date, COALESCE(model, '') AS model, COALESCE(custom_llm_provider, '') AS custom_llm_provider, "
    "SUM(cache_read_input_tokens)::bigint AS cache_read_input_tokens, "
    "SUM(cache_creation_input_tokens)::bigint AS cache_creation_input_tokens, "
    "SUM(compression_saved_tokens)::bigint AS compression_saved_tokens, "
    "SUM(spend)::float AS spend "
    'FROM "LiteLLM_DailyUserSpend" WHERE date >= $1 AND date <= $2{user_filter} '
    "GROUP BY date, COALESCE(model, ''), COALESCE(custom_llm_provider, '') "
    "ORDER BY date"
)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_cost_per_token: float
    cache_read_cost_per_token: float | None
    cache_creation_cost_per_token: float | None


@dataclass(frozen=True, slots=True)
class SavingsAmounts:
    cache_savings: float
    compression_savings: float


class _CostMapEntry(BaseModel):
    input_cost_per_token: float | None = None
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None


class _DailySavingsRow(BaseModel):
    date: str
    model: str
    custom_llm_provider: str
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    compression_saved_tokens: int
    spend: float


class _SpendLogRow(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request_id: str
    startTime: datetime
    model: str
    custom_llm_provider: str | None = None
    total_tokens: int | None = None
    spend: float | None = None
    metadata: object = None


_DAILY_ROWS_ADAPTER = TypeAdapter(list[_DailySavingsRow])


def _pricing_candidates(model: str, custom_llm_provider: str) -> tuple[str, ...]:
    if not custom_llm_provider:
        return (model,)
    return tuple(
        dict.fromkeys(
            (
                f"{custom_llm_provider}/{model}",
                model,
                model.removeprefix(f"{custom_llm_provider}/"),
            )
        )
    )


def resolve_model_pricing(model: str, custom_llm_provider: str, cost_map: Mapping[str, object]) -> ModelPricing | None:
    for candidate in _pricing_candidates(model, custom_llm_provider):
        raw = cost_map.get(candidate)
        if not isinstance(raw, Mapping):
            continue
        try:
            entry = _CostMapEntry.model_validate(dict(raw))
        except ValidationError:
            continue
        if entry.input_cost_per_token:
            return ModelPricing(
                input_cost_per_token=entry.input_cost_per_token,
                cache_read_cost_per_token=entry.cache_read_input_token_cost,
                cache_creation_cost_per_token=entry.cache_creation_input_token_cost,
            )
    return None


def compute_savings_amounts(
    cache_read_tokens: int,
    cache_creation_tokens: int,
    compression_saved_tokens: int,
    pricing: ModelPricing | None,
) -> SavingsAmounts:
    if pricing is None:
        return SavingsAmounts(cache_savings=0.0, compression_savings=0.0)
    read_savings = (
        cache_read_tokens * (pricing.input_cost_per_token - pricing.cache_read_cost_per_token)
        if pricing.cache_read_cost_per_token is not None
        else 0.0
    )
    write_premium = (
        cache_creation_tokens * (pricing.cache_creation_cost_per_token - pricing.input_cost_per_token)
        if pricing.cache_creation_cost_per_token is not None
        else 0.0
    )
    return SavingsAmounts(
        cache_savings=read_savings - write_premium,
        compression_savings=compression_saved_tokens * pricing.input_cost_per_token,
    )


def _is_unpriced(cache_read_tokens: int, compression_saved_tokens: int, pricing: ModelPricing | None) -> bool:
    if pricing is None:
        return cache_read_tokens > 0 or compression_saved_tokens > 0
    return cache_read_tokens > 0 and pricing.cache_read_cost_per_token is None


def _metrics_for_rows(
    rows: list[_DailySavingsRow],
    pricing_by_key: Mapping[tuple[str, str], ModelPricing | None],
) -> CostSavingsMetrics:
    amounts = [
        compute_savings_amounts(
            cache_read_tokens=row.cache_read_input_tokens,
            cache_creation_tokens=row.cache_creation_input_tokens,
            compression_saved_tokens=row.compression_saved_tokens,
            pricing=pricing_by_key[(row.model, row.custom_llm_provider)],
        )
        for row in rows
    ]
    cache_savings = sum(amount.cache_savings for amount in amounts)
    compression_savings = sum(amount.compression_savings for amount in amounts)
    return CostSavingsMetrics(
        cache_savings=cache_savings,
        compression_savings=compression_savings,
        total_savings=cache_savings + compression_savings,
        spend=sum(row.spend for row in rows),
        cache_read_input_tokens=sum(row.cache_read_input_tokens for row in rows),
        cache_creation_input_tokens=sum(row.cache_creation_input_tokens for row in rows),
        compression_saved_tokens=sum(row.compression_saved_tokens for row in rows),
    )


def build_activity_response(
    rows: list[_DailySavingsRow], cost_map: Mapping[str, object]
) -> CostSavingsActivityResponse:
    pricing_by_key = {
        (row.model, row.custom_llm_provider): resolve_model_pricing(row.model, row.custom_llm_provider, cost_map)
        for row in rows
    }
    results = [
        DailyCostSavings(date=date_value, metrics=_metrics_for_rows(list(day_rows), pricing_by_key))
        for date_value, day_rows in groupby(rows, key=lambda row: row.date)
    ]
    unpriced_models = sorted(
        {
            row.model or "(unknown)"
            for row in rows
            if _is_unpriced(
                row.cache_read_input_tokens,
                row.compression_saved_tokens,
                pricing_by_key[(row.model, row.custom_llm_provider)],
            )
        }
    )
    return CostSavingsActivityResponse(
        results=results,
        totals=_metrics_for_rows(rows, pricing_by_key),
        unpriced_models=unpriced_models,
    )


def _parse_request_metadata(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def summarize_optimized_request(row: _SpendLogRow, cost_map: Mapping[str, object]) -> OptimizedRequestSummary | None:
    metadata = _parse_request_metadata(row.metadata)
    usage_object = metadata.get("usage_object")
    usage = usage_object if isinstance(usage_object, dict) else {}
    cache_read_tokens = extract_cache_read_tokens(usage)
    cache_creation_tokens = extract_cache_creation_tokens(usage)
    compression_saved_tokens = extract_compression_saved_tokens(metadata)
    if cache_read_tokens <= 0 and compression_saved_tokens <= 0:
        return None
    pricing = resolve_model_pricing(row.model, row.custom_llm_provider or "", cost_map)
    amounts = compute_savings_amounts(
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        compression_saved_tokens=compression_saved_tokens,
        pricing=pricing,
    )
    savings = amounts.cache_savings + amounts.compression_savings
    optimized_cost = row.spend or 0.0
    optimizations: list[OptimizationType] = [
        *(["caching"] if cache_read_tokens > 0 else []),
        *(["compression"] if compression_saved_tokens > 0 else []),
    ]
    return OptimizedRequestSummary(
        request_id=row.request_id,
        start_time=row.startTime.isoformat(),
        model=row.model,
        total_tokens=row.total_tokens or 0,
        optimizations=optimizations,
        original_cost=optimized_cost + savings,
        optimized_cost=optimized_cost,
        savings=savings,
    )


def _validated_date(value: str, param: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"Invalid {param}: expected YYYY-MM-DD, got {value!r}"},
        ) from e
    return parsed.isoformat()


def _scoped_user_id(user_api_key_dict: UserAPIKeyAuth) -> str | None:
    if _user_has_admin_view(user_api_key_dict):
        return None
    return require_caller_user_id_for_non_admin(user_api_key_dict)


@router.get(
    "/cost_savings/activity",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CostSavingsActivityResponse,
)
@management_endpoint_wrapper
async def get_cost_savings_activity(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[str | None, fastapi.Query(description="Start date in YYYY-MM-DD format")] = None,
    end_date: Annotated[str | None, fastapi.Query(description="End date in YYYY-MM-DD format")] = None,
) -> CostSavingsActivityResponse:
    """
    Daily cost savings from prompt caching and prompt compression over a date window.

    Admins see gateway-wide savings; other callers see savings for their own usage.
    Savings are computed from the daily spend aggregates joined with the model cost
    map at query time; models missing prices are reported in unpriced_models.
    """
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # circular import with proxy_server

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )
    window_start = _validated_date(start_date, "start_date")
    window_end = _validated_date(end_date, "end_date")
    scoped_user_id = _scoped_user_id(user_api_key_dict)
    sql = _ACTIVITY_SQL.format(user_filter=" AND user_id = $3" if scoped_user_id is not None else "")
    params = [window_start, window_end, *([scoped_user_id] if scoped_user_id is not None else [])]
    raw_rows = await prisma_client.db.query_raw(sql, *params)
    rows = _DAILY_ROWS_ADAPTER.validate_python(raw_rows)
    return build_activity_response(rows, litellm.model_cost)


@router.get(
    "/cost_savings/recent_requests",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RecentOptimizedRequestsResponse,
)
@management_endpoint_wrapper
async def get_recent_optimized_requests(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[str | None, fastapi.Query(description="Start date in YYYY-MM-DD format")] = None,
    end_date: Annotated[str | None, fastapi.Query(description="End date in YYYY-MM-DD format")] = None,
    limit: Annotated[int, fastapi.Query(ge=1, le=100)] = 20,
) -> RecentOptimizedRequestsResponse:
    """
    Most recent requests in the window that benefited from prompt caching or
    prompt compression, with their actual cost, counterfactual unoptimized cost,
    and savings.

    Scans up to the scanned_requests most recent spend logs in the window;
    admins see all requests, other callers see their own.
    """
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # circular import with proxy_server

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )
    window_start = datetime.combine(
        date.fromisoformat(_validated_date(start_date, "start_date")), datetime.min.time(), tzinfo=timezone.utc
    )
    window_end_exclusive = datetime.combine(
        date.fromisoformat(_validated_date(end_date, "end_date")) + timedelta(days=1),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    scoped_user_id = _scoped_user_id(user_api_key_dict)
    user_scope = {"user": scoped_user_id} if scoped_user_id is not None else {}
    raw_rows = await prisma_client.db.litellm_spendlogs.find_many(
        where={"startTime": {"gte": window_start, "lt": window_end_exclusive}, **user_scope},
        order={"startTime": "desc"},
        take=RECENT_REQUESTS_SCAN_WINDOW,
    )
    summaries = [
        summary
        for summary in (
            summarize_optimized_request(_SpendLogRow.model_validate(raw_row), litellm.model_cost)
            for raw_row in raw_rows
        )
        if summary is not None
    ]
    return RecentOptimizedRequestsResponse(requests=summaries[:limit], scanned_requests=len(raw_rows))
