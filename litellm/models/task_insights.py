from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel

UNKNOWN_TASK_ID: Final = "unknown"
UNCLASSIFIED_TASK_ID: Final = "unclassified"
TASK_INSIGHTS_ANALYSIS_TAG: Final = "litellm-task-insights-analysis"
MAX_DISCOVERY_REQUESTS: Final = 500

TaskStatus: TypeAlias = Literal["draft", "frozen"]
TaskSource: TypeAlias = Literal["discovered", "sample"]
AssignmentOutcome: TypeAlias = Literal["assigned", "unknown", "failed"]
DiscoveryStatus: TypeAlias = Literal["idle", "requested", "running", "failed", "completed"]
TaskCapability: TypeAlias = Literal["function_calling", "vision", "reasoning", "structured_output"]
PointPlacement: TypeAlias = Literal["embedding", "categorical"]


class _Frozen(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)


class TaskDefinition(_Frozen):
    task_id: str
    name: str
    description: str
    inclusion_criteria: str
    exclusion_criteria: str
    examples: tuple[str, ...] = ()
    required_capabilities: tuple[TaskCapability, ...] = ()
    status: TaskStatus
    source: TaskSource
    map_x: float
    map_y: float
    discovery_size: int = 0
    created_at: datetime
    frozen_at: datetime | None = None


class DraftTaskEdit(_Frozen):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    inclusion_criteria: str | None = Field(default=None, max_length=2000)
    exclusion_criteria: str | None = Field(default=None, max_length=2000)


class TaskInsightsState(_Frozen):
    discovery_status: DiscoveryStatus = "idle"
    discovery_requested_at: datetime | None = None
    discovery_window_start: datetime | None = None
    discovery_window_end: datetime | None = None
    discovery_max_requests: int = MAX_DISCOVERY_REQUESTS
    discovery_finished_at: datetime | None = None
    discovery_error: str | None = None
    discovery_provenance: Mapping[str, str | int | float | None] = Field(default_factory=dict)
    taxonomy_frozen_at: datetime | None = None
    last_bucketed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class InsightsWindow:
    start: datetime
    end: datetime
    team_id: str | None = None


class TaskVolume(_Frozen):
    task_id: str
    request_count: int
    delivery_count: int
    session_count: int
    spend: float | None
    cost_observations: int
    unattributed_count: int = 0


class MatrixCell(_Frozen):
    model_config_id: str
    task_id: str
    request_count: int
    avg_cost_usd: float | None
    cost_observations: int
    p95_latency_ms: float | None
    latency_observations: int


class ObservedPoint(_Frozen):
    request_id: str
    task_id: str
    x: float | None
    y: float | None


class RequestSample(_Frozen):
    request_id: str
    start_time: datetime
    model_config_id: str | None
    model_group: str | None
    session_id: str | None
    spend: float | None
    latency_ms: float | None
    delivery_count: int
    outcome: AssignmentOutcome | None
    top_probability: float | None
    excerpt: str | None


class WindowTotals(_Frozen):
    request_count: int
    delivery_count: int
    session_count: int
    analysis_request_count: int
    analysis_spend: float | None


class Freshness(_Frozen):
    last_observed_at: datetime | None
    last_bucketed_at: datetime | None


class TaskInsightsStore(Protocol):
    async def get_state(self) -> TaskInsightsState: ...

    async def list_definitions(self) -> tuple[TaskDefinition, ...]: ...

    async def task_volumes(self, window: InsightsWindow) -> tuple[TaskVolume, ...]: ...

    async def matrix_cells(self, window: InsightsWindow) -> tuple[MatrixCell, ...]: ...

    async def map_points(self, window: InsightsWindow, limit: int) -> tuple[ObservedPoint, ...]: ...

    async def request_samples(
        self, window: InsightsWindow, task_id: str, model_config_id: str | None, limit: int
    ) -> tuple[RequestSample, ...]: ...

    async def window_totals(self, window: InsightsWindow) -> WindowTotals: ...

    async def freshness(self) -> Freshness: ...

    async def edit_draft(self, task_id: str, edit: DraftTaskEdit) -> TaskDefinition | None: ...

    async def freeze_draft(self, frozen_at: datetime) -> int: ...

    async def request_discovery(self, window: InsightsWindow, max_requests: int, requested_at: datetime) -> bool: ...


class PendingRequest(_Frozen):
    request_id: str
    text_hash: str
    request_text: str
    context_text: str | None


class Decision(_Frozen):
    outcome: AssignmentOutcome
    task_id: str | None
    probabilities: Mapping[str, float] = Field(default_factory=dict)
    top_probability: float | None = None
    fit_probability: float | None = None
    decision_model: str
    provenance: Mapping[str, str | int | float | None] = Field(default_factory=dict)
    error: str | None = None


class DiscoveredTask(_Frozen):
    task_id: str
    name: str
    description: str
    inclusion_criteria: str
    exclusion_criteria: str
    examples: tuple[str, ...]
    required_capabilities: tuple[TaskCapability, ...]
    map_x: float
    map_y: float
    discovery_size: int


class MapPlacement(_Frozen):
    request_id: str
    task_id: str | None
    x: float
    y: float


class DiscoveryRun(_Frozen):
    window: tuple[datetime, datetime]
    max_requests: int


class TaskAnalysisStore(Protocol):
    async def frozen_definitions(self) -> tuple[TaskDefinition, ...]: ...

    async def pending_requests(self, limit: int) -> tuple[PendingRequest, ...]: ...

    async def decisions_by_hash(self, text_hashes: tuple[str, ...]) -> Mapping[str, Decision]: ...

    async def taxonomy_version(self) -> datetime | None: ...

    async def save_decisions(
        self,
        decisions: tuple[tuple[PendingRequest, Decision], ...],
        bucketed_at: datetime,
        taxonomy_version: datetime | None,
    ) -> int: ...

    async def claim_discovery(self) -> DiscoveryRun | None: ...

    async def discovery_samples(self, run: DiscoveryRun) -> tuple[PendingRequest, ...]: ...

    async def save_draft_taxonomy(
        self,
        tasks: tuple[DiscoveredTask, ...],
        placements: tuple[MapPlacement, ...],
        provenance: Mapping[str, str | int | float | None],
        finished_at: datetime,
        source: TaskSource = "discovered",
    ) -> None: ...

    async def fail_discovery(self, error: str, finished_at: datetime) -> None: ...


class UnobservedDelivery(_Frozen):
    delivery_id: str
    request_id: str
    is_batch: bool
    accounting_payload: Mapping[str, object]
    model_config_id: str | None = None
    session_id: str | None = None
    logged_model: str | None = None
    turns: tuple[object, ...] = ()
    input_text: str | None = None


class TaskObservation(_Frozen):
    delivery_id: str
    request_id: str
    is_batch: bool
    start_time: datetime
    team_id: str | None
    model_group: str | None
    model: str | None
    model_config_id: str | None
    session_id: str | None
    spend: float | None
    latency_ms: float | None
    status: str | None
    request_text: str | None
    context_text: str | None
    text_hash: str | None
    excluded: bool
    inference: bool = True
