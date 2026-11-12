from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field

from litellm.models.task_insights import (
    MAX_DISCOVERY_REQUESTS,
    DiscoveryStatus,
    PointPlacement,
    RequestSample,
    TaskCapability,
    TaskDefinition,
)
from litellm.types.llms.base import LiteLLMBaseModel

TaskKind: TypeAlias = Literal["task", "unknown", "unclassified"]
Compatibility: TypeAlias = Literal["compatible", "incompatible", "unknown"]


class _Response(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)


class TaskInsightsWindowOut(_Response):
    start_date: str = Field(description="First day of the measurement window, YYYY-MM-DD (UTC)")
    end_date: str = Field(description="Last day of the measurement window, YYYY-MM-DD (UTC), inclusive")
    team_id: str | None = Field(default=None, description="Team the figures are filtered to, if any")


class TaskInsightsStatus(_Response):
    discovery_status: DiscoveryStatus
    discovery_error: str | None
    discovery_requested_at: datetime | None
    discovery_finished_at: datetime | None
    taxonomy_frozen_at: datetime | None
    last_bucketed_at: datetime | None
    sample_data: bool = Field(description="True when the taxonomy and assignments are seeded sample data")
    draft_task_count: int
    frozen_task_count: int


class TaskSummary(_Response):
    task_id: str
    name: str
    description: str
    kind: TaskKind
    request_count: int
    delivery_count: int
    session_count: int = Field(description="Distinct sessions with a request in this task")
    share: float = Field(description="Fraction of requests in the window, from 0 to 1")
    spend: float | None = Field(description="Sum of known request costs in USD, null when no cost was recorded")
    cost_observations: int = Field(description="Requests whose cost is known")
    avg_cost_usd: float | None
    map_x: float | None = Field(description="Layout position only, distance does not measure model quality")
    map_y: float | None


class TaskMapPoint(_Response):
    request_id: str
    task_id: str
    x: float
    y: float
    placement: PointPlacement = Field(
        description="`embedding` points come from the discovery projection, `categorical` points are drawn at a"
        " stable offset around their task and carry no semantic position"
    )


class TaskInsightsTotals(_Response):
    request_count: int
    delivery_count: int
    session_count: int = Field(description="Distinct sessions in the window, each counted once across tasks")
    spend: float | None
    cost_observations: int
    unattributed_count: int = Field(description="Requests with no delivery tied to a known deployment")
    classified_count: int
    unknown_count: int
    unclassified_count: int
    analysis_request_count: int = Field(description="Requests the analysis worker made, excluded from the figures")
    analysis_spend: float | None


class TaskInsightsOverviewResponse(_Response):
    window: TaskInsightsWindowOut
    last_refreshed_at: datetime | None
    status: TaskInsightsStatus
    totals: TaskInsightsTotals
    tasks: tuple[TaskSummary, ...]
    points: tuple[TaskMapPoint, ...] = Field(description="A bounded deterministic sample of requests for the map")


class ModelConfigSummary(_Response):
    model_config_id: str
    model_name: str | None = Field(description="Public model name of the deployment")
    provider_model: str | None = Field(description="Provider model the deployment calls")
    configured: bool = Field(description="False when traffic names a deployment that is no longer configured")
    max_input_tokens: int | None = Field(
        default=None, description="From model metadata. Whether a given request fits is for the router to check"
    )


class TaskMatrixEntry(_Response):
    model_config_id: str
    task_id: str
    request_count: int
    avg_cost_usd: float | None
    cost_observations: int
    p95_latency_ms: float | None
    latency_observations: int
    eval_success_rate: float | None = Field(default=None, description="Not measured yet, always null")
    eval_sample_count: int = Field(default=0, description="Not measured yet, always 0")


class TaskCompatibility(_Response):
    model_config_id: str
    task_id: str
    status: Compatibility
    missing: tuple[TaskCapability, ...] = Field(
        description="Required capabilities the model metadata says it lacks. A router must still check each request"
    )


class TaskRef(_Response):
    task_id: str
    name: str
    kind: TaskKind
    required_capabilities: tuple[TaskCapability, ...]


class TaskMatrixResponse(_Response):
    measurement_window: TaskInsightsWindowOut
    last_refreshed_at: datetime | None
    tasks: tuple[TaskRef, ...]
    model_configs: tuple[ModelConfigSummary, ...]
    entries: tuple[TaskMatrixEntry, ...]
    compatibility: tuple[TaskCompatibility, ...]
    unattributed_request_count: int = Field(description="Requests whose serving deployment is unknown")


class TaskDetailResponse(_Response):
    window: TaskInsightsWindowOut
    summary: TaskSummary
    definition: TaskDefinition | None
    model_config_id: str | None
    entries: tuple[TaskMatrixEntry, ...]
    model_configs: tuple[ModelConfigSummary, ...]
    requests: tuple[RequestSample, ...]


class TaskTaxonomyResponse(_Response):
    status: TaskInsightsStatus
    definitions: tuple[TaskDefinition, ...]


class TaskDiscoveryRequest(_Response):
    start_date: str | None = Field(default=None, description="YYYY-MM-DD, defaults to 30 days before end_date")
    end_date: str | None = Field(default=None, description="YYYY-MM-DD, defaults to today")
    max_requests: int = Field(default=MAX_DISCOVERY_REQUESTS, ge=20, le=MAX_DISCOVERY_REQUESTS)


class TaskFreezeRequest(_Response):
    draft_task_count: int = Field(
        ge=1, description="Number of draft tasks reviewed. Freezing fails when the draft changed since the review"
    )


class TaskFreezeResponse(_Response):
    frozen_task_count: int
