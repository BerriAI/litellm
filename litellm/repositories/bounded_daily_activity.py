"""Bounded daily totals and optional model contributions for Activity panels."""

from datetime import date
from typing import TYPE_CHECKING, Final, Literal, cast  # noqa: TID251  # generated Prisma transaction interface

from pydantic import BaseModel, TypeAdapter

from litellm.repositories.analytics_budget import AnalyticsDatabase, BoundedAnalyticsDatabase
from litellm.repositories.daily_activity_repository import DailyActivityQueryDatabase
from litellm.repositories.daily_activity_sql import build_bounded_daily_activity_sql
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    BoundedDailyActivityResponse,
    DailyActivityModelBreakdown,
    DailyActivityModelContribution,
    DailyActivityOtherModels,
    DailySpendData,
    DailySpendMetadata,
    SpendMetrics,
)
from litellm.types.repositories.daily_activity import DailyActivityScope

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class _ActivityRow(BaseModel, frozen=True):
    kind: Literal["day", "total", "model", "other"]
    date: date | None
    model: str | None
    model_count: int
    total_api_keys: int | None
    metrics: SpendMetrics


def _metadata(row: _ActivityRow) -> DailySpendMetadata:
    metrics: Final = row.metrics
    return DailySpendMetadata(
        total_spend=metrics.spend,
        total_flat_cost=metrics.flat_cost,
        total_prompt_tokens=metrics.prompt_tokens,
        total_completion_tokens=metrics.completion_tokens,
        total_tokens=metrics.total_tokens,
        total_api_requests=metrics.api_requests,
        total_successful_requests=metrics.successful_requests,
        total_failed_requests=metrics.failed_requests,
        total_cache_read_input_tokens=metrics.cache_read_input_tokens,
        total_cache_creation_input_tokens=metrics.cache_creation_input_tokens,
        total_compression_saved_tokens=metrics.compression_saved_tokens,
        total_compression_savings_spend=metrics.compression_savings_spend,
        total_prompt_caching_savings_spend=metrics.prompt_caching_savings_spend,
        total_gateway_injected_caching_savings_spend=metrics.gateway_injected_caching_savings_spend,
        total_autorouter_savings_spend=metrics.autorouter_savings_spend,
        total_response_time_ms=metrics.total_response_time_ms,
        total_timed_requests=metrics.timed_requests,
        total_api_keys=row.total_api_keys,
    )


class BoundedDailyActivityRepository:
    def __init__(self, database: DailyActivityQueryDatabase) -> None:
        self.database = database

    @classmethod
    def from_prisma(cls, prisma: "PrismaClient") -> "BoundedDailyActivityRepository":
        return cls(BoundedAnalyticsDatabase(cast(AnalyticsDatabase, prisma.db)))  # cast-ok: generated Prisma tx

    async def aggregate(
        self, scope: DailyActivityScope, *, model_limit: int | None = None
    ) -> BoundedDailyActivityResponse:
        if model_limit is not None and not 1 <= model_limit <= 100:
            raise ValueError("model_limit must be between 1 and 100")
        query: Final = build_bounded_daily_activity_sql(scope, model_limit=model_limit)
        rows: Final = TypeAdapter(tuple[_ActivityRow, ...]).validate_python(
            await self.database.query_raw(query.sql, *query.params)
        )
        total: Final = next(row for row in rows if row.kind == "total")
        other: Final = next((row for row in rows if row.kind == "other"), None)
        days: Final = tuple(
            DailySpendData(date=row.date, metrics=row.metrics)
            for row in rows
            if row.kind == "day" and row.date is not None
        )
        models: Final = tuple(
            DailyActivityModelContribution(model=row.model, metrics=row.metrics)
            for row in rows
            if row.kind == "model" and row.model is not None
        )
        return BoundedDailyActivityResponse(
            results=tuple(sorted(days, key=lambda day: day.date, reverse=True)),
            metadata=_metadata(total),
            model_breakdown=DailyActivityModelBreakdown(
                limit=model_limit,
                total_models=total.model_count,
                models=tuple(sorted(models, key=lambda model: (-model.metrics.spend, model.model))),
                other=DailyActivityOtherModels(model_count=other.model_count, metrics=other.metrics) if other else None,
            )
            if model_limit is not None
            else None,
        )
