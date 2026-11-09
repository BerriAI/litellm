import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, JsonValue

from litellm.proxy.workflows.monty.models import PythonProgram
from litellm.types.llms.base import LiteLLMBaseModel


class NodeType(str, Enum):
    CHAT_COMPLETION = "chat_completion"
    PYTHON = "python"


class BindingOutput(str, Enum):
    MESSAGES = "messages"
    RESULT = "result"


class Reference(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    source: str = Field(alias="from", min_length=1, max_length=128)
    output: BindingOutput
    path: str | None = Field(default=None, max_length=512)


class Constant(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    value: JsonValue


Binding: TypeAlias = Reference | Constant


class RetryPolicy(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=1, ge=1, le=3)
    backoff_seconds: float = Field(default=0, ge=0, le=10, allow_inf_nan=False)


class ResponseSchema(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    strict: bool = True
    schema_: Mapping[str, JsonValue] = Field(alias="schema")


class StructuredResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["json_schema"] = "json_schema"
    json_schema: ResponseSchema


class ChatCompletionNode(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    type: Literal[NodeType.CHAT_COMPLETION]
    model: str = Field(min_length=1, max_length=256)
    messages: Binding
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stop: str | tuple[str, ...] | None = None
    response_format: StructuredResponse | None = None
    timeout_seconds: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)
    retry: RetryPolicy = RetryPolicy()


class PythonNode(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    type: Literal[NodeType.PYTHON]
    program: PythonProgram
    inputs: Mapping[str, Binding]
    timeout_seconds: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)
    retry: RetryPolicy = RetryPolicy()


WorkflowNode: TypeAlias = Annotated[ChatCompletionNode | PythonNode, Field(discriminator="type")]


class WorkflowBundle(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: Literal["1"]
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,62}$")
    nodes: tuple[WorkflowNode, ...] = Field(min_length=1, max_length=64)
    timeout_seconds: float = Field(default=300, gt=0, le=900, allow_inf_nan=False)
    input_schema: Mapping[str, JsonValue]
    output: Reference


class WorkflowFailure(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    param: str | None = None
    status_code: int = 400


def canonical_bundle(bundle: WorkflowBundle) -> str:
    return json.dumps(
        bundle.model_dump(mode="json", by_alias=True, exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def workflow_digest(bundle: WorkflowBundle) -> str:
    return hashlib.sha256(canonical_bundle(bundle).encode()).hexdigest()


def reference_failure(reference: Reference, available: frozenset[str]) -> WorkflowFailure | None:
    if reference.source == "$input" and reference.output is not BindingOutput.MESSAGES:
        return WorkflowFailure(code="invalid_reference", message="Request input exposes only messages")
    if reference.source != "$input" and (
        reference.source not in available or reference.output is not BindingOutput.RESULT
    ):
        return WorkflowFailure(code="invalid_reference", message="Reference must select a completed predecessor")
    if reference.path is not None and not reference.path.startswith("/") and reference.path != "":
        return WorkflowFailure(code="invalid_reference", message="path must be a JSON Pointer")
    return None


def validate_bindings(bundle: WorkflowBundle) -> WorkflowFailure | None:
    if bundle.name == "runs":
        return WorkflowFailure(code="reserved_name", message="runs is reserved for run tracking")
    ids: Final = tuple(node.id for node in bundle.nodes)
    if len(set(ids)) != len(ids):
        return WorkflowFailure(code="duplicate_node", message="Node IDs must be unique")
    for index, node in enumerate(bundle.nodes):
        bindings = (node.messages,) if isinstance(node, ChatCompletionNode) else tuple(node.inputs.values())
        for binding in bindings:
            failure = reference_failure(binding, frozenset(ids[:index])) if isinstance(binding, Reference) else None
            if failure is not None:
                return failure.model_copy(update={"param": f"nodes.{node.id}"})
        if isinstance(node, ChatCompletionNode) and node.model.startswith("workflow/"):
            return WorkflowFailure(
                code="nested_workflow", message="Nested workflow calls are unsupported", param=node.id
            )
        if isinstance(node, PythonNode) and set(node.inputs) != set(node.program.parameter_names):
            return WorkflowFailure(code="invalid_inputs", message="Bind exactly the function parameters", param=node.id)
    return reference_failure(bundle.output, frozenset(ids))
