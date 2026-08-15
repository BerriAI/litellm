from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import (
    Required,
    TypedDict,
)

from .openai import ChatCompletionUsageBlock


class GenericStreamingChunk(TypedDict, total=False):
    text: Required[str]
    is_finished: Required[bool]
    finish_reason: Required[str | None]
    logprobs: BaseModel | None
    original_chunk: BaseModel | None
    usage: BaseModel | None


class DatabricksTextContent(TypedDict, total=False):
    type: Literal["text"]
    text: Required[str]
    citations: list[dict[str, Any]] | None


class DatabricksReasoningSummary(TypedDict):
    type: Literal["summary_text"]
    text: str
    signature: str


class DatabricksReasoningContent(TypedDict, total=False):
    type: Literal["reasoning"]
    summary: Required[list[DatabricksReasoningSummary]]
    citations: list[dict[str, Any]] | None


AllDatabricksContentListValues = DatabricksTextContent | DatabricksReasoningContent

AllDatabricksContentValues = str | list[AllDatabricksContentListValues]


class DatabricksFunction(TypedDict, total=False):
    name: Required[str]
    description: dict | str
    parameters: dict
    strict: bool


class DatabricksTool(TypedDict):
    function: DatabricksFunction
    type: Literal["function"]


class DatabricksMessage(TypedDict, total=False):
    role: Required[str]
    content: Required[AllDatabricksContentValues]
    tool_calls: list[DatabricksTool] | None


class DatabricksChoice(TypedDict, total=False):
    index: Required[int]
    message: Required[DatabricksMessage]
    finish_reason: Required[str | None]
    extra_fields: str


class DatabricksResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[DatabricksChoice]
    usage: ChatCompletionUsageBlock
