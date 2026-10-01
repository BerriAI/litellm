from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Scope(Record):
    team_id: str = ""
    api_key_hash: str = ""
    all_teams: bool = False


class MetadataFilter(Record):
    key: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=500)


class Check(Record):
    id: str = Field(min_length=1, max_length=80)
    instruction: str = Field(min_length=3, max_length=3000)
    enabled: bool = True


class EngineSettings(Record):
    name: str = Field(min_length=1, max_length=100)
    context: str = Field(default="", max_length=6000)
    source: Literal["traces", "requests", "both"] = "traces"
    lookback_hours: int = Field(default=24, ge=1, le=720)
    service: str = Field(default="", max_length=200)
    filters: tuple[MetadataFilter, ...] = Field(default=(), max_length=8)
    checks: tuple[Check, ...] = Field(min_length=1, max_length=12)
    model: str = Field(min_length=1, max_length=200)
    enabled: bool = True
    interval_minutes: int = Field(default=15, ge=1, le=10080)
    sample_size: int = Field(default=100, ge=1, le=500)
    monthly_budget: float = Field(default=20, gt=0, le=100000, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique_checks(self) -> "EngineSettings":
        if len(frozenset(c.id for c in self.checks)) != len(self.checks):
            raise ValueError("Each check must have a unique ID")
        return self


class Evidence(Record):
    execution_id: str
    span_id: str
    quote: str = Field(min_length=1, max_length=1000)


class FindingDraft(Record):
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=10, max_length=4000)
    check_id: str
    kind: Literal["issue", "pattern"] = "issue"
    priority: Literal["high", "medium", "low"] = "medium"
    suggestion: str = Field(default="", max_length=2000)
    limitation: str = Field(default="", max_length=600)
    evidence: tuple[Evidence, ...] = Field(min_length=1, max_length=20)
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


class Job(Record):
    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    stage: str = "Queued"
    created_at: datetime
    start: datetime
    end: datetime
    settings: EngineSettings
    revision: int
    worker_id: str | None = None
    lease_until: datetime | None = None
    attempts: int = 0
    finished_at: datetime | None = None
    coverage: Coverage = Coverage()
    error: str = ""
    sample: Sample | None = None
    cost: float = 0


class Engine(Record):
    id: str
    scope: Scope
    settings: EngineSettings
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
    id: str
    name: str
    scope: Scope
    last_seen: datetime
    revoked: bool = False


class WorkerCreated(Record):
    worker: Worker
    token: str


class EngineList(Record):
    engines: tuple[Engine, ...]
    workers: tuple[Worker, ...]
    tracing_enabled: bool


class RunRequest(Record):
    lookback_hours: int | None = Field(default=None, ge=1, le=720)


class FindingUpdate(Record):
    status: Literal["open", "resolved", "dismissed"]
    reason: str = Field(default="", max_length=2000)


class Claim(Record):
    engine_id: str
    job: Job
    findings: tuple[Finding, ...]


class Progress(Record):
    stage: str = Field(max_length=100)
    coverage: Coverage = Coverage()


class Result(Record):
    findings: tuple[FindingDraft, ...] = Field(default=(), max_length=30)
    coverage: Coverage
    error: str = Field(default="", max_length=1000)


class ModelRequest(Record):
    prompt: str = Field(min_length=1, max_length=100000)
    purpose: Literal["extract", "cluster", "investigate"]


class ModelResult(Record):
    content: str
    cost: float
