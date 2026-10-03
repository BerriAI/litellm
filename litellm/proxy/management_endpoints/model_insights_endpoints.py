from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from litellm.constants import MODEL_INSIGHTS_DEFAULT_TASK, MODEL_INSIGHTS_MAX_RANGE_DAYS, MODEL_INSIGHTS_TOP_MODELS
from litellm.llms.oss_decision import OSS_DECISION_MODELS
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.model_insights_tasks import load_model_insight_tasks
from litellm.proxy.db.model_usage_task_classifier import (
    MODEL_USAGE_TASK_CLASSIFIER,
    MODEL_USAGE_TASK_CLASSIFIER_PARAM,
    ModelUsageTaskClassifier,
    build_model_usage_task_classifier_client,
)
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.table_repositories import DailyModelUsageRepository
from litellm.router_strategy.complexity_router.config import OpenSourceClassifierConfig
from litellm.router_strategy.complexity_router.jev_classifier import JevClassifierClient
from litellm.secret_managers.main import get_secret_str
from litellm.types.model_insights import (
    ModelInsightDailyMetric,
    ModelInsightDailyTotal,
    ModelInsightMetric,
    ModelInsightsMetric,
    ModelInsightsResponse,
    ModelInsightTask,
    ModelInsightTaskClassifierConfig,
    ModelInsightTaskClassifierProvider,
    ModelInsightTaskClassifierProviderName,
    ModelInsightTaskClassifierResponse,
    ModelInsightTasksResponse,
    ModelInsightTaskSummary,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

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


class _GroupedDate(BaseModel):
    date: str
    sums: _Sums = Field(alias="_sum")


class _GroupedTask(_GroupedModel):
    task_type: str


_MODEL_ROWS: Final = TypeAdapter(list[_GroupedModel])
_DAILY_ROWS: Final = TypeAdapter(list[_GroupedDaily])
_DATE_ROWS: Final = TypeAdapter(list[_GroupedDate])
_TASK_ROWS: Final = TypeAdapter(list[_GroupedTask])
_UNCATEGORIZED_TASK: Final = ModelInsightTask(
    task_type=MODEL_INSIGHTS_DEFAULT_TASK,
    label="Uncategorized",
    category="General",
    description="Requests without a recognized task classification.",
)
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


def _rank_value(row: _GroupedModel, metric: ModelInsightsMetric) -> float:
    if metric == "requests":
        return row.sums.request_count
    if metric == "spend":
        return row.sums.spend
    return row.sums.prompt_tokens + row.sums.completion_tokens


def _top_model_rows(rows: list[_GroupedModel], metric: ModelInsightsMetric) -> list[_GroupedModel]:
    return sorted(rows, key=lambda row: _rank_value(row, metric), reverse=True)[:MODEL_INSIGHTS_TOP_MODELS]


def _deployment_filter(rows: list[_GroupedModel]) -> list[dict[str, str]]:
    return [
        {"model_group": row.model_group, "model": row.model, "custom_llm_provider": row.custom_llm_provider}
        for row in rows
    ]


def _daily_metric(row: _GroupedDaily) -> ModelInsightDailyMetric:
    return ModelInsightDailyMetric(date=row.date, **_metric(row).model_dump())


def _daily_total(row: _GroupedDate) -> ModelInsightDailyTotal:
    return ModelInsightDailyTotal(
        date=row.date,
        spend=row.sums.spend,
        prompt_tokens=row.sums.prompt_tokens,
        completion_tokens=row.sums.completion_tokens,
        requests=row.sums.request_count,
    )


def _summarize_tasks(rows: list[_GroupedTask], metric: ModelInsightsMetric) -> list[ModelInsightTaskSummary]:
    catalog: Final = load_model_insight_tasks()
    first_seen: Final = {task: index for index, task in enumerate(dict.fromkeys(row.task_type for row in rows))}
    by_task: Final = {
        task: tuple(group)
        for task, group in itertools.groupby(
            sorted(rows, key=lambda row: first_seen[row.task_type]), key=lambda row: row.task_type
        )
    }
    totals: Final = {
        task: functools.reduce(lambda total, row: total + _rank_value(row, metric), task_rows, 0.0)
        for task, task_rows in by_task.items()
    }
    leaders: Final = {
        task: max(task_rows, key=lambda row: _rank_value(row, metric)) for task, task_rows in by_task.items()
    }
    grand: Final = sum(totals.values())
    return [
        ModelInsightTaskSummary(
            **(catalog.get(task) or _UNCATEGORIZED_TASK).model_copy(update={"task_type": task}).model_dump(),
            value=value,
            share=value / grand * 100 if grand else 0.0,
            leader=leaders[task].model_group,
            provider=leaders[task].custom_llm_provider,
        )
        for task, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def _resolve_window(
    user_api_key_dict: UserAPIKeyAuth, start_date: str | None, end_date: str | None
) -> tuple[date, date, Mapping[str, object], DailyModelUsageRepository]:
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        raise HTTPException(status_code=403, detail="Only proxy admins can view deployment-wide model insights")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    end_day: Final = _parse_date(end_date, datetime.now(timezone.utc).date())
    start_day: Final = _parse_date(start_date, end_day - timedelta(days=MODEL_INSIGHTS_MAX_RANGE_DAYS - 1))
    if start_day > end_day or (end_day - start_day).days >= MODEL_INSIGHTS_MAX_RANGE_DAYS:
        raise HTTPException(
            status_code=400, detail=f"Date range must be between 1 and {MODEL_INSIGHTS_MAX_RANGE_DAYS} days"
        )
    date_window: Final[Mapping[str, object]] = {"date": {"gte": start_day.isoformat(), "lte": end_day.isoformat()}}
    return start_day, end_day, date_window, DailyModelUsageRepository(prisma_client)


def _model_insights_task_classifier_environment_lookup() -> Callable[[str], str | None]:
    return get_secret_str


def _model_insights_task_classifier_client_builder() -> Callable[[OpenSourceClassifierConfig], JevClassifierClient]:
    return build_model_usage_task_classifier_client


def _model_insights_task_classifier_runtime() -> ModelUsageTaskClassifier:
    return MODEL_USAGE_TASK_CLASSIFIER


def _classifier_required_environment(provider: ModelInsightTaskClassifierProviderName) -> str:
    match provider:
        case "jev":
            return "TYPESAFE_API_KEY"
        case "laya":
            return "LAYA_API_BASE"
        case "bespoke":
            return "BESPOKE_API_BASE"


def _classifier_models(provider: ModelInsightTaskClassifierProviderName) -> tuple[str, ...]:
    match provider:
        case "jev":
            return ("jev-latest",)
        case "laya":
            return OSS_DECISION_MODELS["laya"]
        case "bespoke":
            return OSS_DECISION_MODELS["bespoke"]


def _classifier_provider(
    provider: ModelInsightTaskClassifierProviderName,
    label: str,
    env_lookup: Callable[[str], str | None],
) -> ModelInsightTaskClassifierProvider:
    required_env: Final = _classifier_required_environment(provider)
    ready: Final = bool(env_lookup(required_env))
    return ModelInsightTaskClassifierProvider(
        provider=provider,
        label=label,
        models=list(_classifier_models(provider)),
        ready=ready,
        missing_env=[] if ready else [required_env],
    )


def _task_classifier_response(
    configured: ModelInsightTaskClassifierConfig | None,
    env_lookup: Callable[[str], str | None],
) -> ModelInsightTaskClassifierResponse:
    return ModelInsightTaskClassifierResponse(
        configured=configured,
        providers=[
            _classifier_provider("jev", "Jev (TypeSafe)", env_lookup),
            _classifier_provider("laya", "Laya", env_lookup),
            _classifier_provider("bespoke", "Bespoke Nimble", env_lookup),
        ],
    )


def _task_classifier_config(stored_value: object | None) -> ModelInsightTaskClassifierConfig | None:
    if stored_value is None:
        return None
    try:
        return ModelInsightTaskClassifierConfig.model_validate(stored_value)
    except ValidationError:
        return None


def _require_task_classifier_role(user: UserAPIKeyAuth, *, allow_view_only: bool) -> None:
    allowed_roles: Final = (
        (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        if allow_view_only
        else (LitellmUserRoles.PROXY_ADMIN,)
    )
    if user.user_role not in allowed_roles:
        raise HTTPException(status_code=403, detail="Only proxy admins can manage the model insights task classifier")


def _model_insights_task_classifier_prisma() -> PrismaClient:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    return prisma_client


@router.get(
    "/model-insights/task-classifier",
    tags=["model insights"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ModelInsightTaskClassifierResponse,
)
async def get_model_insights_task_classifier(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    env_lookup: Annotated[
        Callable[[str], str | None],
        Depends(_model_insights_task_classifier_environment_lookup),
    ],
) -> ModelInsightTaskClassifierResponse:
    _require_task_classifier_role(user_api_key_dict, allow_view_only=True)
    prisma_client: Final = _model_insights_task_classifier_prisma()
    stored: Final = await ConfigRepository(prisma_client).get_param(MODEL_USAGE_TASK_CLASSIFIER_PARAM)
    return _task_classifier_response(_task_classifier_config(stored.param_value if stored is not None else None), env_lookup)


@router.put(
    "/model-insights/task-classifier",
    tags=["model insights"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ModelInsightTaskClassifierResponse,
)
async def put_model_insights_task_classifier(
    body: ModelInsightTaskClassifierConfig,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    env_lookup: Annotated[
        Callable[[str], str | None],
        Depends(_model_insights_task_classifier_environment_lookup),
    ],
    client_builder: Annotated[
        Callable[[OpenSourceClassifierConfig], JevClassifierClient],
        Depends(_model_insights_task_classifier_client_builder),
    ],
    classifier: Annotated[ModelUsageTaskClassifier, Depends(_model_insights_task_classifier_runtime)],
) -> ModelInsightTaskClassifierResponse:
    _require_task_classifier_role(user_api_key_dict, allow_view_only=False)
    prisma_client: Final = _model_insights_task_classifier_prisma()
    required_env: Final = _classifier_required_environment(body.provider)
    if not env_lookup(required_env):
        raise HTTPException(
            status_code=400,
            detail=f"Provider {body.provider!r} is not ready; set {required_env}",
        )
    if body.model not in _classifier_models(body.provider):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model {body.model!r} for provider {body.provider!r}",
        )
    try:
        classifier_config: Final = OpenSourceClassifierConfig(provider=body.provider, model=body.model)
        client: Final = client_builder(classifier_config)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid task classifier configuration: {exc}") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not configure provider {body.provider!r} ({type(exc).__name__})",
        ) from exc
    await ConfigRepository(prisma_client).set_param(
        MODEL_USAGE_TASK_CLASSIFIER_PARAM,
        body.model_dump(mode="json"),
    )
    await classifier.activate(body, client)
    return _task_classifier_response(body, env_lookup)


@router.delete(
    "/model-insights/task-classifier",
    tags=["model insights"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ModelInsightTaskClassifierResponse,
)
async def delete_model_insights_task_classifier(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    env_lookup: Annotated[
        Callable[[str], str | None],
        Depends(_model_insights_task_classifier_environment_lookup),
    ],
    classifier: Annotated[ModelUsageTaskClassifier, Depends(_model_insights_task_classifier_runtime)],
) -> ModelInsightTaskClassifierResponse:
    _require_task_classifier_role(user_api_key_dict, allow_view_only=False)
    prisma_client: Final = _model_insights_task_classifier_prisma()
    await ConfigRepository(prisma_client).delete_param(MODEL_USAGE_TASK_CLASSIFIER_PARAM)
    await classifier.clear(prisma_client)
    return _task_classifier_response(None, env_lookup)


@router.get(
    "/model-insights",
    tags=["model insights"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ModelInsightsResponse,
)
async def get_model_insights(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[str | None, Query(description="YYYY-MM-DD, defaults to 365 days ago")] = None,
    end_date: Annotated[str | None, Query(description="YYYY-MM-DD, defaults to today")] = None,
    metric: Annotated[ModelInsightsMetric, Query(description="Metric the top models are ranked by")] = "tokens",
) -> ModelInsightsResponse:
    start_day, end_day, date_window, repository = _resolve_window(user_api_key_dict, start_date, end_date)
    table: Final = repository.table
    grouped_model_rows: Final = _MODEL_ROWS.validate_python(
        await table.group_by(
            by=["model_group", "model", "custom_llm_provider"],
            sum=_SUM_FIELDS,
            where=date_window,
        )
    )
    model_rows: Final = _top_model_rows(grouped_model_rows, metric)
    selected_window: Final = {**date_window, "OR": _deployment_filter(model_rows)}
    daily_rows: Final = _DAILY_ROWS.validate_python(
        await table.group_by(
            by=["date", "model_group", "model", "custom_llm_provider"],
            sum=_SUM_FIELDS,
            where=selected_window,
            order={"date": "asc"},
        )
        if model_rows
        else []
    )
    date_rows: Final = _DATE_ROWS.validate_python(
        await table.group_by(
            by=["date"],  # mutable-ok: prisma group_by requires a list of fields
            sum=_SUM_FIELDS,
            where=date_window,
            order={"date": "asc"},  # mutable-ok: prisma order clause must be a dict
        )
    )
    return ModelInsightsResponse(
        start_date=start_day.isoformat(),
        end_date=end_day.isoformat(),
        top_models=[_metric(row) for row in model_rows],
        daily=[_daily_metric(row) for row in daily_rows],
        daily_totals=tuple(_daily_total(row) for row in date_rows),
    )


@router.get(
    "/model-insights/tasks",
    tags=["model insights"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ModelInsightTasksResponse,
)
async def get_model_insight_tasks(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[str | None, Query(description="YYYY-MM-DD, defaults to 365 days ago")] = None,
    end_date: Annotated[str | None, Query(description="YYYY-MM-DD, defaults to today")] = None,
    metric: Annotated[ModelInsightsMetric, Query(description="Metric task shares are computed from")] = "spend",
) -> ModelInsightTasksResponse:
    start_day, end_day, date_window, repository = _resolve_window(user_api_key_dict, start_date, end_date)
    task_rows: Final = _TASK_ROWS.validate_python(
        await repository.table.group_by(
            by=["task_type", "model_group", "model", "custom_llm_provider"],
            sum=_SUM_FIELDS,
            where=date_window,
        )
    )
    return ModelInsightTasksResponse(
        start_date=start_day.isoformat(),
        end_date=end_day.isoformat(),
        tasks=_summarize_tasks(task_rows, metric),
    )
