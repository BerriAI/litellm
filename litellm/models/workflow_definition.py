from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Protocol, TypeAlias

from liteflow import (  # pyright: ignore[reportMissingTypeStubs]  # pinned SDK omits its py.typed marker
    Draft as WorkflowDraft,
)
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy.workflows.monty.models import PythonProgram

WorkflowName: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")]
Digest: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ValueBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    value: JsonValue


class ReferenceBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    source: str = Field(alias="from", min_length=1, max_length=128)
    output: Literal["messages", "result"]


Binding: TypeAlias = ValueBinding | ReferenceBinding


class WorkflowRetry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_attempts: int = Field(default=1, ge=1, le=3, strict=True)
    backoff_seconds: float = Field(default=0, ge=0, le=30)


class StructuredSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    name: str = Field(min_length=1, max_length=128)
    strict: Literal[True]
    schema_: Mapping[str, JsonValue] = Field(alias="schema")


class StructuredOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["json_schema"]
    json_schema: StructuredSchema


class ChatCompletionNode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    id: WorkflowName
    type: Literal["chat_completion"]
    model: str = Field(min_length=1, max_length=256)
    messages: Binding
    timeout_seconds: float = Field(gt=0, le=300)
    retry: WorkflowRetry
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0, strict=True)
    max_completion_tokens: int | None = Field(default=None, gt=0, strict=True)
    seed: int | None = Field(default=None, strict=True)
    stop: str | tuple[str, ...] | None = None
    response_format: StructuredOutput | None = None


class PythonNode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    id: WorkflowName
    type: Literal["python"]
    program: PythonProgram
    inputs: Mapping[str, Binding]
    timeout_seconds: float = Field(gt=0, le=300)
    retry: WorkflowRetry


WorkflowNode: TypeAlias = Annotated[ChatCompletionNode | PythonNode, Field(discriminator="type")]


class WorkflowBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    contract_version: Literal["1"]
    name: WorkflowName
    timeout_seconds: float = Field(gt=0, le=1800)
    input_schema: Mapping[str, JsonValue]
    output: ReferenceBinding
    nodes: tuple[WorkflowNode, ...] = Field(min_length=1, max_length=64)


def parsed_bundle(bundle: Mapping[str, JsonValue]) -> WorkflowBundle:
    return WorkflowBundle.model_validate(bundle)


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    draft: WorkflowDraft
    contributors: tuple[str, ...]


class WorkflowSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str
    workflow_id: str
    team_id: str
    name: str
    digest: Digest
    bundle: WorkflowBundle
    contributors: tuple[str, ...]
    published_by: str | None
    created_at: datetime


class DefinitionPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflows: tuple[WorkflowDraft, ...]
    next_cursor: str | None


DefinitionErrorCode: TypeAlias = Literal[
    "not_found", "forbidden", "team_required", "invalid_bundle", "conflict", "stale_draft", "precondition_required"
]


@dataclass(frozen=True, slots=True)
class DefinitionError:
    code: DefinitionErrorCode
    message: str
    param: str | None = None


class WorkflowDefinitionStore(Protocol):
    async def get_team(self, team_id: str) -> LiteLLM_TeamTable | None: ...

    async def get_draft(self, workflow_id: str, team_id: str | None = None) -> WorkflowDefinition | None: ...

    async def list_drafts(self, team_id: str, limit: int, cursor: str | None) -> DefinitionPage: ...

    async def get_snapshot(self, snapshot_id: str, team_id: str | None = None) -> WorkflowSnapshot | None: ...

    async def get_published(self, name: str, team_id: str) -> WorkflowSnapshot | None: ...

    async def create(
        self, bundle: WorkflowBundle, team_id: str, contributor: str
    ) -> WorkflowDraft | DefinitionError: ...

    async def replace(
        self, workflow_id: str, team_id: str, etag: str, bundle: WorkflowBundle, contributor: str
    ) -> WorkflowDraft | DefinitionError: ...

    async def publish(
        self, workflow_id: str, team_id: str, etag: str, digest: str, publisher: str
    ) -> WorkflowSnapshot | DefinitionError: ...

    async def freeze_draft(self, workflow_id: str, team_id: str, etag: str) -> WorkflowSnapshot | DefinitionError: ...
