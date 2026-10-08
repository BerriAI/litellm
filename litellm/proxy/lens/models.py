import json
from datetime import datetime, timedelta, timezone
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    model_validator,
)


def calendar_lookback(hours: int) -> int:
    try:
        datetime.now(timezone.utc) - timedelta(hours=hours)
    except OverflowError as error:
        raise ValueError("Lookback exceeds the supported calendar range") from error
    return hours


def calendar_interval(minutes: int) -> int:
    try:
        datetime.now(timezone.utc) + timedelta(minutes=minutes)
    except OverflowError as error:
        raise ValueError("Interval exceeds the supported calendar range") from error
    return minutes


LookbackHours: TypeAlias = Annotated[int, Field(ge=1), AfterValidator(calendar_lookback)]
IntervalMinutes: TypeAlias = Annotated[int, Field(ge=1), AfterValidator(calendar_interval)]


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Scope(Record):
    team_id: str = ""
    api_key_hash: str = ""
    all_teams: bool = False


class MetadataFilter(Record):
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)


class Check(Record):
    id: str = Field(min_length=1)
    instruction: str = Field(min_length=3)
    enabled: bool = True


class ActivitySelection(Record):
    source: Literal["traces", "requests", "both"] = "traces"
    service: str = Field(default="")
    agent_name: str = Field(default="")
    filters: tuple[MetadataFilter, ...] = Field(default=())
    sample_size: int | None = Field(default=None, ge=1)
    sample_percent: float = Field(default=100, gt=0, le=100, allow_inf_nan=False)
    team_id: str = ""
    execution_ids: tuple[str, ...] = ()


class LensSettings(ActivitySelection):
    name: str = Field(min_length=1)
    context: str = Field(default="")
    lookback_hours: LookbackHours = 24
    checks: tuple[Check, ...] = ()
    model: str = Field(min_length=1)
    enabled: bool = True
    interval_minutes: IntervalMinutes = 15
    concurrency: int = Field(default=8, ge=1)
    monthly_budget: float = Field(default=100, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique_checks(self) -> "LensSettings":
        if len(frozenset(c.id for c in self.checks)) != len(self.checks):
            raise ValueError("Each check must have a unique ID")
        if not self.context.strip() and not any(c.enabled for c in self.checks):
            raise ValueError("Describe expected behavior or add an enabled check")
        if any(c.id == "expected_behavior" for c in self.checks):
            raise ValueError("expected_behavior is reserved for the behavior description")
        return self

    @property
    def analysis_checks(self) -> tuple[Check, ...]:
        behavior: Final = (
            (
                Check(
                    id="expected_behavior",
                    instruction="Identify deviations from the expected behavior described in context.",
                ),
            )
            if self.context.strip()
            else ()
        )
        return (*behavior, *(c for c in self.checks if c.enabled))


class Evidence(Record):
    execution_id: str
    span_id: str
    quote: str = Field(min_length=1)
    role: Literal["support", "counterexample"] = "support"


class Observation(Record):
    check_id: str
    kind: Literal["issue", "pattern"] = "issue"
    summary: str
    evidence: tuple[Evidence, ...] = ()


class Extraction(Record):
    observations: tuple[Observation, ...] = ()
    cannot_assess: bool = False
    reasoning: str = Field(default="", max_length=800)


class ReviewVersion(Record):
    execution_id: str
    content_version: str


class AgentTestCase(Record):
    input: str = Field(min_length=1)
    expected: str = Field(min_length=1)


class IssueBrief(Record):
    problem: str = Field(min_length=10)
    user_goal: str = Field(min_length=3)
    what_happened: str = Field(min_length=3)
    test_cases: tuple[AgentTestCase, ...] = Field(min_length=1)


class FindingDraft(Record):
    title: str = Field(min_length=3)
    description: str = Field(min_length=10)
    check_id: str
    kind: Literal["issue", "pattern"] = "issue"
    priority: Literal["high", "medium", "low"] = "medium"
    suggestion: str = Field(default="")
    limitation: str = Field(default="")
    brief: IssueBrief | None = None
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    existing_finding_id: str | None = None
    check_ids: tuple[str, ...] = ()
    merged_finding_ids: tuple[str, ...] = ()


class Finding(FindingDraft):
    id: str
    status: Literal["open", "resolved", "dismissed"] = "open"
    reason: str = ""
    first_seen: datetime
    last_seen: datetime
    occurrences: tuple[str, ...] = ()
    revision: int
    investigation_runs: tuple[str, ...] = ()


class Coverage(Record):
    eligible: int = 0
    selected: int = 0
    screened: int = 0
    investigated: int = 0
    inconclusive: int = 0
    grouping_batches: int = 0
    grouped_batches: int = 0
    candidates: int = 0
    partial: int = 0
    unassessable: int = 0
    failed_tasks: int = Field(default=0, ge=0)
    reused: int = Field(default=0, ge=0)
    reusable: int = Field(default=0, ge=0)


class Execution(Record):
    id: str
    source: Literal["traces", "requests"]
    trace_id: str
    trace_ref: str = ""
    team_id: str
    name: str
    start_time: str
    span_count: int
    root_seen: bool = False
    service: str = ""
    metadata: tuple[MetadataFilter, ...] = ()


class TracePart(Record):
    execution_id: str
    span_id: str
    parent_span_id: str = ""
    name: str
    kind: str
    content: str
    truncated: bool = False
    start_time: str = ""
    end_time: str = ""


class ExecutionContent(Record):
    execution: Execution
    parts: tuple[TracePart, ...]
    next_cursor: str | None = None
    partial: bool = False


class Sample(Record):
    executions: tuple[Execution, ...]
    eligible: int
    selected: int = 0
    next_offset: int | None = None
    next_cursor: str | None = None


class RunAssessment(Record):
    execution_id: str
    issue_checks: tuple[str, ...] = ()
    pattern_checks: tuple[str, ...] = ()
    cannot_assess: bool = False


class TraceIdentity(Record):
    trace_id: str = Field(min_length=1, max_length=128)
    trace_ref: str = Field(default="", max_length=512)


class TraceFindingsRequest(Record):
    traces: tuple[TraceIdentity, ...] = Field(min_length=1, max_length=500)


class TraceFindingCount(TraceIdentity):
    finding_count: int | None = Field(ge=0)


MAX_STEPS = 200


class Step(Record):
    at: datetime
    kind: Literal["stage", "model", "error"]
    label: str = Field(max_length=200)
    model: str = Field(default="", max_length=200)
    purpose: str = Field(default="", max_length=40)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0


MAX_REVIEWS = 60


ActivityOperation: TypeAlias = Literal[
    "model",
    "read",
    "search",
    "python",
    "catalog",
    "review_catalog",
    "read_reviews",
    "search_reviews",
    "history",
    "checkpoint",
]
ActivityPhase: TypeAlias = Literal["load", "review", "group", "reconcile", "investigate"]


class ToolCount(Record):
    name: ActivityOperation
    calls: int = Field(ge=0)


class Activity(Record):
    id: str
    phase: ActivityPhase
    label: str
    execution_ids: tuple[str, ...] = ()
    started_at: datetime
    operations: tuple[ActivityOperation, ...] = ()
    tool_calls: tuple[ToolCount, ...] = ()
    finished: bool = False


class ReviewSpan(Record):
    span_id: str
    name: str = Field(max_length=120)
    kind: str = Field(max_length=40)
    preview: str = Field(max_length=240)
    cited: bool = False


class ReviewVerdict(Record):
    check_id: str
    kind: Literal["issue", "pattern"]
    summary: str = Field(max_length=300)


class Review(Record):
    execution_id: str
    trace_id: str
    agent: str
    name: str
    spans: tuple[ReviewSpan, ...] = Field(default=(), max_length=8)
    reasoning: str = Field(default="", max_length=800)
    verdicts: tuple[ReviewVerdict, ...] = ()
    cannot_assess: bool = False
    model: str
    duration_ms: int = Field(ge=0)
    at: datetime
    tool_calls: tuple[ToolCount, ...] = ()
    extraction: Extraction | None = None
    content_version: str = ""
    reused: bool = False
    consolidated: bool = False
    partial: bool = False


class ReviewPage(Record):
    reviews: tuple[Review, ...]
    reviewed: int


class InFlight(Record):
    execution_id: str
    trace_id: str
    agent: str
    started_at: datetime


class Job(Record):
    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    stage: str = "Queued"
    created_at: datetime
    start: datetime
    end: datetime
    settings: LensSettings
    revision: int
    worker_id: str | None = None
    lease_until: datetime | None = None
    attempts: int = 0
    finished_at: datetime | None = None
    coverage: Coverage = Coverage()
    error: str = ""
    sample: Sample | None = None
    cost: float = 0
    findings: tuple[Finding, ...] | None = None
    assessments: tuple[RunAssessment, ...] = ()
    steps: tuple[Step, ...] = ()
    reviews: tuple[Review, ...] = ()
    reviewed: int = 0
    reading: tuple[InFlight, ...] = ()
    activities: tuple[Activity, ...] = ()
    trigger: Literal["schedule", "manual"] = "schedule"
    review_versions: tuple[ReviewVersion, ...] = ()


class BudgetReservation(Record):
    id: str
    job_id: str
    amount: float = Field(ge=0, allow_inf_nan=False)
    month: str
    expires_at: datetime | None = None


class Lens(Record):
    id: str
    scope: Scope
    settings: LensSettings
    revision: int = 1
    version: int = 0
    created_at: datetime
    next_run_at: datetime
    last_scan_at: datetime | None = None
    jobs: tuple[Job, ...] = ()
    findings: tuple[Finding, ...] = ()
    budget_month: str
    spent: float = 0
    reservations: tuple[BudgetReservation, ...] = ()
    criteria_updated_at: datetime | None = None


class Worker(Record):
    analysis_key_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    id: str
    name: str
    scope: Scope
    last_seen: datetime
    revoked: bool = False


class WorkerCreated(Record):
    image: str
    worker: Worker
    token: str
    managed: bool = False


class LensList(Record):
    lenses: tuple[Lens, ...]
    workers: tuple[Worker, ...]
    tracing_enabled: bool


class RunRequest(Record):
    settings: LensSettings | None = None
    lookback_hours: LookbackHours | None = None
    start: datetime | None = None
    end: datetime | None = None
    agent_name: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def ordered_window(self) -> "RunRequest":
        if (self.start is None) != (self.end is None):
            raise ValueError("Choose both a start and an end time")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("Start time must be before end time")
        return self


class WatchSkipped(Record):
    id: str
    name: str
    reason: str


class WatchAllResult(Record):
    watching: tuple[str, ...]
    skipped: tuple[WatchSkipped, ...] = ()


class FindingUpdate(Record):
    status: Literal["open", "resolved", "dismissed"]
    reason: str = Field(default="")


class Claim(Record):
    lens_id: str
    job: Job
    findings: tuple[Finding, ...]
    reviews: tuple[Review, ...] | None = None


class Progress(Record):
    stage: str | None = None
    coverage: Coverage | None = None
    review: Review | None = None
    reading: tuple[InFlight, ...] | None = None
    activity: Activity | None = None


class Result(Record):
    assessments: tuple[RunAssessment, ...] = ()
    findings: tuple[FindingDraft, ...] = ()
    coverage: Coverage
    error: str = Field(default="")
    review_versions: tuple[ReviewVersion, ...] = ()


class ModelMessage(Record):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelRequest(Record):
    prompt: str = Field(min_length=1)
    purpose: Literal["extract", "cluster", "investigate"]
    messages: tuple[ModelMessage, ...] = ()

    def conversation(self) -> tuple[ModelMessage, ...]:
        if self.messages:
            return self.messages
        try:
            payload: Final = TypeAdapter(dict[str, JsonValue]).validate_json(self.prompt)
        except ValidationError:
            if self.prompt.lstrip().startswith(("{", "[")):
                raise ValueError("Malformed legacy Lens prompt; send structured messages.") from None
            return (ModelMessage(role="system", content=self.prompt), ModelMessage(role="user", content="{}"))
        instruction_fields: Final = frozenset(
            ("task", "navigation", "context", "checks", "questions", "response_schema")
        )
        instructions: Final = {key: value for key, value in payload.items() if key in instruction_fields}
        evidence: Final = {key: value for key, value in payload.items() if key not in instruction_fields}
        return (
            ModelMessage(role="system", content=json.dumps(instructions, ensure_ascii=False)),
            ModelMessage(role="user", content=json.dumps(evidence, ensure_ascii=False)),
        )


class ModelResult(Record):
    content: str
    cost: float
    context_exceeded: bool = False
    finish_reason: Literal["length", "content_filter"] | None = Field(default=None, exclude=True)


class DatasetToolCall(Record):
    name: str
    arguments: str


class DatasetMessage(Record):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str = ""
    tool_calls: tuple[DatasetToolCall, ...] = ()


class CaseSource(Record):
    trace_id: str = ""
    trace_ref: str = ""
    span_id: str = ""
    finding_id: str = ""
    lens_id: str = ""


class DatasetCase(Record):
    id: str
    messages: tuple[DatasetMessage, ...]
    reply: str = ""
    tool_calls: tuple[DatasetToolCall, ...] = ()
    expected: str = ""
    included: bool = True
    source: CaseSource
    agent_version: str = ""


class SkippedCase(Record):
    source: CaseSource
    reason: Literal["duplicate", "no_content", "too_large", "over_limit", "invalid"]


class TraceSource(Record):
    kind: Literal["trace"] = "trace"
    trace_id: str = Field(min_length=1)
    trace_ref: str = ""
    span_id: str = ""


class FindingSource(Record):
    kind: Literal["finding"] = "finding"
    lens_id: str = Field(min_length=1)
    finding_ids: tuple[str, ...] = Field(min_length=1)


class TextSource(Record):
    kind: Literal["text"] = "text"
    text: str = Field(min_length=1)


BuildSource: TypeAlias = Annotated[TraceSource | FindingSource | TextSource, Field(discriminator="kind")]


class BuildRequest(Record):
    sources: tuple[BuildSource, ...] = Field(min_length=1)
    dataset_id: str = ""


class BuildResult(Record):
    cases: tuple[DatasetCase, ...]
    skipped: tuple[SkippedCase, ...]


class DatasetCreate(Record):
    name: str = Field(min_length=1, max_length=120)
    agent_name: str = ""


class Dataset(Record):
    id: str
    name: str
    agent_name: str
    team_id: str
    created_at: datetime
    revision: int
    created_by: str
    cases: tuple[DatasetCase, ...]


class DatasetSummary(Record):
    id: str
    name: str
    agent_name: str
    revision: int
    case_count: int
    updated_at: datetime


class RevisionSave(Record):
    base_revision: int = Field(ge=0)
    cases: tuple[DatasetCase, ...]


class EvalCases(Record):
    dataset_id: str
    revision: int
    cases: tuple[DatasetCase, ...]
