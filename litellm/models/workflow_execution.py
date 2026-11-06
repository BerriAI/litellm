from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class RunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[Mapping[str, JsonValue], ...] = Field(min_length=1, max_length=256)
    end_user_id: str | None = None
    deadline: datetime
    draft: bool


class EngineRun(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    run_id: str
    session_id: str
    workflow_id: str
    snapshot_id: str
    team_id: str
    created_by: str
    status: RunState
    created_at: datetime
    updated_at: datetime
    input: RunInput
    output: JsonValue = None
    metadata: Mapping[str, JsonValue] | None = None
    request_digest: str
    cancel_requested: bool = False
    lease_token: str | None = None
    lease_expires_at: datetime | None = None


class StepTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str
    run_id: str
    node_id: str
    state: RunState
    input: JsonValue
    output: JsonValue = None
    attempts: int = Field(ge=0)
    deadline: datetime
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_cost: float | None = None
    request_id: str | None = None
    call_id: str | None = None
    request_ids: tuple[str, ...] = ()


class RunSteps(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    snapshot_id: str
    state: RunState
    cancel_requested: bool
    steps: tuple[StepTrace, ...]
    error: str | None = None


class RunView(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    run_id: str
    session_id: str
    workflow_id: str
    snapshot_id: str
    team_id: str
    status: RunState
    created_at: datetime
    updated_at: datetime
    cancel_requested: bool


class RunPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    runs: tuple[RunView, ...]
    count: int
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class ChildCallResult:
    output: JsonValue
    prompt_tokens: int
    completion_tokens: int
    request_id: str | None
    response_cost: float | None


@dataclass(frozen=True, slots=True)
class ExecutionError:
    code: str
    message: str
    receipt: ChildCallResult | None = None


def workflow_call_id(run_id: str, node_id: str, attempt: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"litellm/workflows/{run_id}/{node_id}/{attempt}"))


@dataclass(frozen=True, slots=True)
class RunAdmission:
    run: EngineRun
    created: bool


class WorkflowExecutionStore(Protocol):
    async def find_idempotent(
        self, workflow_id: str, team_id: str, key_hash: str, idempotency_key: str
    ) -> EngineRun | None: ...

    async def admit(
        self,
        *,
        workflow_id: str,
        snapshot_id: str,
        team_id: str,
        key_hash: str,
        input: RunInput,
        digest: str,
        idempotency_key: str | None,
    ) -> RunAdmission | ExecutionError: ...

    async def get(self, run_id: str) -> EngineRun | None: ...

    async def recover(self, limit: int) -> tuple[EngineRun, ...]: ...

    async def claim(self, run_id: str, token: str, lease_expires_at: datetime) -> EngineRun | None: ...

    async def renew(self, run_id: str, token: str, lease_expires_at: datetime) -> bool: ...

    async def checkpoint(self, run_id: str, token: str, step: StepTrace) -> bool: ...

    async def steps(self, run_id: str) -> tuple[StepTrace, ...]: ...

    async def finish(self, run_id: str, token: str, state: RunState, output: JsonValue, error: str | None) -> bool: ...

    async def cancel(self, run_id: str) -> None: ...

    async def list_runs(
        self,
        workflow_id: str,
        team_id: str,
        key_hash: str | None,
        state: RunState | None,
        limit: int,
        cursor: str | None,
    ) -> RunPage | ExecutionError: ...
