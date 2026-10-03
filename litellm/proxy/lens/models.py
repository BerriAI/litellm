from datetime import datetime, timedelta, timezone
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


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


class LensSettings(Record):
    name: str = Field(min_length=1)
    context: str = Field(default="")
    source: Literal["traces", "requests", "both"] = "traces"
    lookback_hours: LookbackHours = 24
    service: str = Field(default="")
    agent_name: str = Field(default="")
    filters: tuple[MetadataFilter, ...] = Field(default=())
    checks: tuple[Check, ...] = ()
    model: str = Field(min_length=1)
    enabled: bool = True
    interval_minutes: IntervalMinutes = 15
    sample_size: int | None = Field(default=None, ge=1)
    sample_percent: float = Field(default=100, gt=0, le=100, allow_inf_nan=False)
    concurrency: int = Field(default=8, ge=1)
    team_id: str = ""
    execution_ids: tuple[str, ...] = ()
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


class Finding(FindingDraft):
    id: str
    status: Literal["open", "resolved", "dismissed"] = "open"
    reason: str = ""
    first_seen: datetime
    last_seen: datetime
    occurrences: tuple[str, ...] = ()
    revision: int


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


class Worker(Record):
    analysis_key_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    id: str
    name: str
    scope: Scope
    last_seen: datetime
    revoked: bool = False


class WorkerCreated(Record):
    worker: Worker
    token: str


class LensList(Record):
    lenses: tuple[Lens, ...]
    workers: tuple[Worker, ...]
    tracing_enabled: bool


class RunRequest(Record):
    settings: LensSettings | None = None
    lookback_hours: LookbackHours | None = None


class FindingUpdate(Record):
    status: Literal["open", "resolved", "dismissed"]
    reason: str = Field(default="")


class Claim(Record):
    lens_id: str
    job: Job
    findings: tuple[Finding, ...]


class Progress(Record):
    stage: str = Field()
    coverage: Coverage = Coverage()


class Result(Record):
    assessments: tuple[RunAssessment, ...] = ()
    findings: tuple[FindingDraft, ...] = ()
    coverage: Coverage
    error: str = Field(default="")


class ModelRequest(Record):
    prompt: str = Field(min_length=1)
    purpose: Literal["extract", "cluster", "investigate"]


class ModelResult(Record):
    content: str
    cost: float
    finish_reason: Literal["length", "content_filter"] | None = Field(default=None, exclude=True)
