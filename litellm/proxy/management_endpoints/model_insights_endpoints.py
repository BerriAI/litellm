from collections.abc import Mapping
from datetime import date, timedelta
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, TypeAdapter

from litellm.constants import MODEL_INSIGHTS_TOP_MODELS
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.table_repositories import DailyModelUsageRepository
from litellm.types.model_insights import (
    ModelInsightDailyMetric,
    ModelInsightMetric,
    ModelInsightsResponse,
    ModelInsightTaskMetric,
)

router: Final = APIRouter()


class _Sums(BaseModel):
    spend: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_count: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class _GroupedModel(BaseModel):
    model_group: str
    model: str
    custom_llm_provider: str
    sums: _Sums = Field(alias="_sum")


class _GroupedDaily(_GroupedModel):
    date: str


class _GroupedTask(_GroupedModel):
    task_type: str


_MODEL_ROWS: Final = TypeAdapter(list[_GroupedModel])
_DAILY_ROWS: Final = TypeAdapter(list[_GroupedDaily])
_TASK_ROWS: Final = TypeAdapter(list[_GroupedTask])
_SUM_FIELDS: Final = {
    "spend": True,
    "prompt_tokens": True,
    "completion_tokens": True,
    "request_count": True,
    "successful_requests": True,
    "failed_requests": True,
}


def _parse_date(value: str | None, fallback: date) -> date:
    if value is None:
        return fallback
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Dates must use YYYY-MM-DD") from exc


def _metric(row: _GroupedModel) -> ModelInsightMetric:
    return ModelInsightMetric(
        model_group=row.model_group,
        model=row.model,
        provider=row.custom_llm_provider,
        spend=row.sums.spend,
        prompt_tokens=row.sums.prompt_tokens,
        completion_tokens=row.sums.completion_tokens,
        requests=row.sums.request_count,
        successful_requests=row.sums.successful_requests,
        failed_requests=row.sums.failed_requests,
    )


def _total_tokens(row: _GroupedModel) -> int:
    return row.sums.prompt_tokens + row.sums.completion_tokens


def _top_model_rows(rows: list[_GroupedModel]) -> list[_GroupedModel]:
    return sorted(rows, key=_total_tokens, reverse=True)[:MODEL_INSIGHTS_TOP_MODELS]


def _daily_metric(row: _GroupedDaily) -> ModelInsightDailyMetric:
    return ModelInsightDailyMetric(date=row.date, **_metric(row).model_dump())


def _task_metric(row: _GroupedTask) -> ModelInsightTaskMetric:
    return ModelInsightTaskMetric(task_type=row.task_type, **_metric(row).model_dump())


@router.get(
    "/model-insights",
    tags=["model insights"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ModelInsightsResponse,
)
async def get_model_insights(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[str | None, Query(description="YYYY-MM-DD, defaults to 20 days ago")] = None,
    end_date: Annotated[str | None, Query(description="YYYY-MM-DD, defaults to today")] = None,
) -> ModelInsightsResponse:
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        raise HTTPException(status_code=403, detail="Only proxy admins can view deployment-wide model insights")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    end_day: Final = _parse_date(end_date, date.today())
    start_day: Final = _parse_date(start_date, end_day - timedelta(days=19))
    if start_day > end_day or (end_day - start_day).days > 89:
        raise HTTPException(status_code=400, detail="Date range must be between 1 and 90 days")

    date_window: Final[Mapping[str, object]] = {"date": {"gte": start_day.isoformat(), "lte": end_day.isoformat()}}
    table: Final = DailyModelUsageRepository(prisma_client).table
    grouped_model_rows: Final = _MODEL_ROWS.validate_python(
        await table.group_by(
            by=["model_group", "model", "custom_llm_provider"],
            sum=_SUM_FIELDS,
            where=date_window,
        )
    )
    model_rows: Final = _top_model_rows(grouped_model_rows)
    selected_models: Final = [row.model_group for row in model_rows]
    selected_window: Final = {**date_window, "model_group": {"in": selected_models}}
    daily_rows: Final = _DAILY_ROWS.validate_python(
        await table.group_by(
            by=["date", "model_group", "model", "custom_llm_provider"],
            sum=_SUM_FIELDS,
            where=selected_window,
            order={"date": "asc"},
        )
        if selected_models
        else []
    )
    task_rows: Final = _TASK_ROWS.validate_python(
        await table.group_by(
            by=["task_type", "model_group", "model", "custom_llm_provider"],
            sum=_SUM_FIELDS,
            where=selected_window,
            order={"_sum": {"completion_tokens": "desc"}},
        )
        if selected_models
        else []
    )
    return ModelInsightsResponse(
        start_date=start_day.isoformat(),
        end_date=end_day.isoformat(),
        top_models=[_metric(row) for row in model_rows],
        daily=[_daily_metric(row) for row in daily_rows],
        by_task=[_task_metric(row) for row in task_rows],
    )
