from datetime import datetime
from enum import Enum
from typing import Protocol

from pydantic import ConfigDict, JsonValue

from litellm.types.llms.base import LiteLLMBaseModel


class WorkflowRunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowSnapshotOrigin(str, Enum):
    PUBLICATION = "publication"
    DRAFT_TEST = "draft_test"


class WorkflowDraft(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    team_id: str
    name: str
    etag: str
    bundle_json: str
    contributors: tuple[str, ...]
    published_snapshot_id: str | None


class WorkflowSnapshot(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str
    workflow_id: str
    team_id: str
    name: str
    bundle_json: str
    published_by: str
    created_at: datetime
    origin: WorkflowSnapshotOrigin = WorkflowSnapshotOrigin.PUBLICATION


class WorkflowCheckpoint(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node_path: str
    input: JsonValue
    output: JsonValue
    attempts: int


class WorkflowStep(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str
    run_id: str
    node_id: str
    state: WorkflowRunState
    input: JsonValue
    output: JsonValue | None = None
    attempts: int
    deadline: datetime
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_cost: float | None = None
    request_id: str | None = None


class WorkflowExecution(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    snapshot_id: str
    team_id: str
    name: str
    principal: str
    dispatch_key: str
    request_digest: str
    request_json: str
    state: WorkflowRunState
    deadline: datetime
    lease_owner: str | None = None
    lease_until: datetime | None = None
    cancel_requested: bool
    checkpoints: tuple[WorkflowCheckpoint, ...] = ()
    response_json: str | None = None
    error: str | None = None


class WorkflowConflict(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    status_code: int = 409


class WorkflowStore(Protocol):
    async def create(
        self, team_id: str, name: str, bundle_json: str, author: str
    ) -> WorkflowDraft | WorkflowConflict: ...

    async def get(self, team_id: str, name: str) -> WorkflowDraft | None: ...

    async def get_id(self, workflow_id: str) -> WorkflowDraft | None: ...

    async def list(self, team_id: str, after: str | None, limit: int) -> tuple[WorkflowDraft, ...]: ...

    async def replace(
        self, team_id: str, name: str, etag: str, bundle_json: str, author: str
    ) -> WorkflowDraft | WorkflowConflict: ...

    async def publish(
        self, team_id: str, name: str, etag: str, publisher: str
    ) -> WorkflowSnapshot | WorkflowConflict: ...

    async def snapshot(self, snapshot_id: str) -> WorkflowSnapshot | None: ...

    async def pin_draft(
        self, workflow_id: str, etag: str, requested_by: str
    ) -> WorkflowSnapshot | WorkflowConflict: ...

    async def published(self, team_id: str, name: str) -> WorkflowSnapshot | None: ...

    async def dispatch(self, execution: WorkflowExecution) -> WorkflowExecution: ...

    async def by_dispatch_key(self, dispatch_key: str) -> WorkflowExecution | None: ...

    async def execution(self, run_id: str) -> WorkflowExecution | None: ...

    async def lease_next(self, owner: str) -> WorkflowExecution | None: ...

    async def renew(self, run_id: str, owner: str) -> bool: ...

    async def steps(self, run_id: str) -> tuple[WorkflowStep, ...]: ...

    async def start_step(
        self, run_id: str, node_id: str, input: JsonValue, deadline: datetime, owner: str
    ) -> WorkflowStep | WorkflowConflict: ...

    async def attempt_step(self, step_id: str, owner: str) -> WorkflowStep | WorkflowConflict: ...

    async def finish_step(self, step: WorkflowStep, owner: str) -> bool: ...

    async def checkpoint(self, run_id: str, checkpoint: WorkflowCheckpoint, step: WorkflowStep, owner: str) -> bool: ...

    async def finish(
        self,
        run_id: str,
        state: WorkflowRunState,
        response_json: str | None = None,
        error: str | None = None,
        *,
        owner: str,
    ) -> bool: ...

    async def cancel(self, run_id: str) -> None: ...
