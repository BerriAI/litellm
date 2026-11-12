"""
TASK INSIGHTS

GET   /task-insights/overview            - task volume, spend and the task map for a window
GET   /task-insights/matrix              - model deployment x task metrics
GET   /task-insights/tasks/{task_id}     - one task's definition, metrics and recent requests
GET   /task-insights/taxonomy            - draft and frozen task definitions
PATCH /task-insights/tasks/{task_id}     - edit a draft definition before freezing it
POST  /task-insights/taxonomy/freeze     - replace the frozen taxonomy with the draft
POST  /task-insights/discovery           - queue task discovery for the analysis worker

The figures are deployment-wide and include request text, so reads are limited to proxy admin roles and writes to
proxy admins. No route calls a model, the analysis worker does that out of band.
"""

from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from typing import Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.models.task_insights import (
    UNCLASSIFIED_TASK_ID,
    UNKNOWN_TASK_ID,
    DraftTaskEdit,
    InsightsWindow,
    TaskDefinition,
    TaskInsightsState,
    TaskInsightsStore,
)
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import TaskInsightsRepositoryDep
from litellm.task_insights.summaries import (
    CapabilityProbe,
    compatibility,
    map_points,
    matrix_entries,
    model_capabilities,
    model_configs,
    task_refs,
    task_summaries,
    task_summary,
    totals,
)
from litellm.types.proxy.task_insights import (
    ModelConfigSummary,
    TaskDetailResponse,
    TaskDiscoveryRequest,
    TaskFreezeRequest,
    TaskFreezeResponse,
    TaskInsightsOverviewResponse,
    TaskInsightsStatus,
    TaskInsightsWindowOut,
    TaskMatrixResponse,
    TaskTaxonomyResponse,
)

router: Final = APIRouter()

_DEFAULT_DAYS: Final = 30
_MAX_RANGE_DAYS: Final = 366
_MAP_POINT_LIMIT: Final = 600
_REQUEST_SAMPLE_LIMIT: Final = 25
_TAGS: Final = ["task insights"]
_READ_ROLES: Final = frozenset({LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY})


class _Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    model: str | None = None


class _Info(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str | None = None


class _Deployment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    model_name: str | None = None
    litellm_params: _Params = _Params()
    model_info: _Info = _Info()


_DEPLOYMENTS: Final = TypeAdapter(tuple[_Deployment, ...])


def get_model_catalog() -> tuple[ModelConfigSummary, ...]:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return ()
    try:
        deployments: Final = _DEPLOYMENTS.validate_python(tuple(llm_router.registry().deployments() or ()))
    except ValidationError:
        return ()
    return tuple(
        ModelConfigSummary(
            model_config_id=deployment.model_info.id,
            model_name=deployment.model_name,
            provider_model=deployment.litellm_params.model,
            configured=True,
        )
        for deployment in deployments
        if deployment.model_info.id
    )


def get_capability_probe() -> CapabilityProbe:
    return model_capabilities


def get_clock() -> Callable[[], datetime]:
    return lambda: datetime.now(timezone.utc)


ModelCatalogDep: TypeAlias = Annotated[tuple[ModelConfigSummary, ...], Depends(get_model_catalog)]
CapabilityProbeDep: TypeAlias = Annotated[CapabilityProbe, Depends(get_capability_probe)]
ClockDep: TypeAlias = Annotated[Callable[[], datetime], Depends(get_clock)]
StartDate: TypeAlias = Annotated[str | None, Query(description="YYYY-MM-DD (UTC), defaults to 30 days before end_date")]
EndDate: TypeAlias = Annotated[str | None, Query(description="YYYY-MM-DD (UTC), inclusive, defaults to today")]
TeamId: TypeAlias = Annotated[str | None, Query(description="Only count requests made by this team")]


def _store(user: UserAPIKeyAuth, store: TaskInsightsStore | None, *, write: bool = False) -> TaskInsightsStore:
    allowed: Final = frozenset({LitellmUserRoles.PROXY_ADMIN}) if write else _READ_ROLES
    if user.user_role not in allowed:
        action: Final = "change" if write else "view"
        raise HTTPException(status_code=403, detail=f"Only proxy admins can {action} deployment-wide task insights")
    if store is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    return store


def _day(value: str | None, fallback: date) -> date:
    if value is None:
        return fallback
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Dates must use YYYY-MM-DD") from exc


def _window(
    start_date: str | None, end_date: str | None, team_id: str | None, now: datetime
) -> tuple[InsightsWindow, TaskInsightsWindowOut]:
    end_day: Final = _day(end_date, now.astimezone(timezone.utc).date())
    start_day: Final = _day(start_date, end_day - timedelta(days=_DEFAULT_DAYS - 1))
    if start_day > end_day or (end_day - start_day).days >= _MAX_RANGE_DAYS:
        raise HTTPException(status_code=400, detail=f"Date range must be between 1 and {_MAX_RANGE_DAYS} days")
    window: Final = InsightsWindow(
        start=datetime.combine(start_day, time.min, tzinfo=timezone.utc),
        end=datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=timezone.utc),
        team_id=team_id or None,
    )
    return window, TaskInsightsWindowOut(
        start_date=start_day.isoformat(), end_date=end_day.isoformat(), team_id=window.team_id
    )


def _status(state: TaskInsightsState, definitions: tuple[TaskDefinition, ...]) -> TaskInsightsStatus:
    frozen: Final = tuple(task for task in definitions if task.status == "frozen")
    return TaskInsightsStatus(
        discovery_status=state.discovery_status,
        discovery_error=state.discovery_error,
        discovery_requested_at=state.discovery_requested_at,
        discovery_finished_at=state.discovery_finished_at,
        taxonomy_frozen_at=state.taxonomy_frozen_at,
        last_bucketed_at=state.last_bucketed_at,
        sample_data=bool(frozen) and all(task.source == "sample" for task in frozen),
        draft_task_count=sum(1 for task in definitions if task.status == "draft"),
        frozen_task_count=len(frozen),
    )


def _latest(*moments: datetime | None) -> datetime | None:
    known: Final = tuple(moment for moment in moments if moment is not None)
    return max(known) if known else None


@router.get("/task-insights/overview", tags=_TAGS, response_model=TaskInsightsOverviewResponse)
async def get_task_insights_overview(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: TaskInsightsRepositoryDep,
    clock: ClockDep,
    start_date: StartDate = None,
    end_date: EndDate = None,
    team_id: TeamId = None,
) -> TaskInsightsOverviewResponse:
    """Task volume, spend and a bounded sample of map points. Counts are requests, never repeated deliveries."""
    store: Final = _store(user_api_key_dict, repository)
    window, window_out = _window(start_date, end_date, team_id, clock())
    definitions: Final = await store.list_definitions()
    volumes: Final = await store.task_volumes(window)
    freshness: Final = await store.freshness()
    return TaskInsightsOverviewResponse(
        window=window_out,
        last_refreshed_at=_latest(freshness.last_observed_at, freshness.last_bucketed_at),
        status=_status(await store.get_state(), definitions),
        totals=totals(volumes, await store.window_totals(window)),
        tasks=task_summaries(definitions, volumes),
        points=map_points(await store.map_points(window, _MAP_POINT_LIMIT), definitions),
    )


@router.get("/task-insights/matrix", tags=_TAGS, response_model=TaskMatrixResponse)
async def get_task_insights_matrix(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: TaskInsightsRepositoryDep,
    catalog: ModelCatalogDep,
    probe: CapabilityProbeDep,
    clock: ClockDep,
    start_date: StartDate = None,
    end_date: EndDate = None,
    team_id: TeamId = None,
) -> TaskMatrixResponse:
    """Request count, mean cost and p95 latency per deployment and task. Missing cost or latency stays null and
    `*_observations` says how many requests the figure rests on. Compatibility comes from current model metadata."""
    store: Final = _store(user_api_key_dict, repository)
    window, window_out = _window(start_date, end_date, team_id, clock())
    definitions: Final = await store.list_definitions()
    volumes: Final = await store.task_volumes(window)
    cells: Final = await store.matrix_cells(window)
    configs: Final = model_configs(catalog, cells, probe)
    freshness: Final = await store.freshness()
    return TaskMatrixResponse(
        measurement_window=window_out,
        last_refreshed_at=_latest(freshness.last_observed_at, freshness.last_bucketed_at),
        tasks=task_refs(definitions, volumes),
        model_configs=configs,
        entries=matrix_entries(cells),
        compatibility=compatibility(definitions, configs, probe),
        unattributed_request_count=sum(volume.unattributed_count for volume in volumes),
    )


@router.get("/task-insights/tasks/{task_id}", tags=_TAGS, response_model=TaskDetailResponse)
async def get_task_insights_task(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: TaskInsightsRepositoryDep,
    catalog: ModelCatalogDep,
    probe: CapabilityProbeDep,
    clock: ClockDep,
    task_id: Annotated[str, Path(description="A frozen task id, `unknown` or `unclassified`")],
    start_date: StartDate = None,
    end_date: EndDate = None,
    team_id: TeamId = None,
    model_config_id: Annotated[str | None, Query(description="Only show requests served by this deployment")] = None,
) -> TaskDetailResponse:
    """A task's definition, its metrics per deployment and its most recent requests with bounded text excerpts."""
    store: Final = _store(user_api_key_dict, repository)
    window, window_out = _window(start_date, end_date, team_id, clock())
    definitions: Final = await store.list_definitions()
    definition: Final = next((task for task in definitions if task.task_id == task_id), None)
    if task_id not in {UNKNOWN_TASK_ID, UNCLASSIFIED_TASK_ID} and (definition is None or definition.status != "frozen"):
        raise HTTPException(status_code=404, detail="No frozen task with this id")
    cells: Final = tuple(cell for cell in await store.matrix_cells(window) if cell.task_id == task_id)
    return TaskDetailResponse(
        window=window_out,
        summary=task_summary(task_id, definitions, await store.task_volumes(window)),
        definition=definition,
        model_config_id=model_config_id,
        entries=matrix_entries(cells),
        model_configs=model_configs(catalog, cells, probe),
        requests=await store.request_samples(window, task_id, model_config_id, _REQUEST_SAMPLE_LIMIT),
    )


@router.get("/task-insights/taxonomy", tags=_TAGS, response_model=TaskTaxonomyResponse)
async def get_task_insights_taxonomy(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: TaskInsightsRepositoryDep,
) -> TaskTaxonomyResponse:
    """Draft and frozen task definitions with the discovery state, for reviewing a draft before freezing it."""
    store: Final = _store(user_api_key_dict, repository)
    definitions: Final = await store.list_definitions()
    return TaskTaxonomyResponse(status=_status(await store.get_state(), definitions), definitions=definitions)


@router.patch("/task-insights/tasks/{task_id}", tags=_TAGS, response_model=TaskDefinition)
async def edit_task_insights_draft(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: TaskInsightsRepositoryDep,
    task_id: Annotated[str, Path(description="A draft task id")],
    edit: DraftTaskEdit,
) -> TaskDefinition:
    """Frozen definitions are immutable, so the assignments made against them stay explainable."""
    store: Final = _store(user_api_key_dict, repository, write=True)
    updated: Final = await store.edit_draft(task_id, edit)
    if updated is None:
        raise HTTPException(status_code=404, detail="No draft task with this id")
    return updated


@router.post("/task-insights/taxonomy/freeze", tags=_TAGS, response_model=TaskFreezeResponse)
async def freeze_task_insights_taxonomy(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: TaskInsightsRepositoryDep,
    clock: ClockDep,
    body: TaskFreezeRequest,
) -> TaskFreezeResponse:
    """Replaces the frozen taxonomy with the draft and clears every assignment, which the worker then rebuilds."""
    store: Final = _store(user_api_key_dict, repository, write=True)
    drafts: Final = sum(1 for task in await store.list_definitions() if task.status == "draft")
    if drafts != body.draft_task_count:
        raise HTTPException(
            status_code=409, detail="The draft changed since it was reviewed, reload and review it again"
        )
    frozen: Final = await store.freeze_draft(clock())
    if not frozen:
        raise HTTPException(status_code=409, detail="There is no draft taxonomy to freeze")
    return TaskFreezeResponse(frozen_task_count=frozen)


@router.post("/task-insights/discovery", tags=_TAGS, response_model=TaskTaxonomyResponse)
async def request_task_insights_discovery(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    repository: TaskInsightsRepositoryDep,
    clock: ClockDep,
    body: TaskDiscoveryRequest,
) -> TaskTaxonomyResponse:
    """Queues discovery over up to `max_requests` distinct request texts. The analysis worker embeds and clusters
    them and drafts task definitions for review, which is the only step that calls models."""
    store: Final = _store(user_api_key_dict, repository, write=True)
    now: Final = clock()
    window, _ = _window(body.start_date, body.end_date, None, now)
    if not await store.request_discovery(window, body.max_requests, now):
        raise HTTPException(status_code=409, detail="Discovery is already queued or running")
    definitions: Final = await store.list_definitions()
    return TaskTaxonomyResponse(status=_status(await store.get_state(), definitions), definitions=definitions)
