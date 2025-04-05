import json
from typing import Any, List, Literal, Optional, TypedDict, Union

from pydantic import BaseModel
from typing_extensions import (
    Protocol,
    Required,
    Self,
    TypeGuard,
    get_origin,
    override,
    runtime_checkable,
)

from .openai import ChatCompletionToolCallChunk, ChatCompletionUsageBlock


class GenericStreamingChunk(TypedDict, total=False):
    text: Required[str]
    is_finished: Required[bool]
    finish_reason: Required[Optional[str]]
    logprobs: Optional[BaseModel]
    original_chunk: Optional[BaseModel]
    usage: Optional[BaseModel]


class DatabricksTextContent(TypedDict):
    type: Literal["text"]
    text: Required[str]


class DatabricksReasoningSummary(TypedDict):
    type: Literal["summary_text"]
    text: str
    signature: str


class DatabricksReasoningContent(TypedDict):
    type: Literal["reasoning"]
    summary: List[DatabricksReasoningSummary]


AllDatabricksContentListValues = Union[
    DatabricksTextContent, DatabricksReasoningContent
]

AllDatabricksContentValues = Union[str, List[AllDatabricksContentListValues]]


class DatabricksFunction(TypedDict, total=False):
    name: Required[str]
    description: dict
    parameters: dict
    strict: bool


class DatabricksTool(TypedDict):
    function: DatabricksFunction
    type: Literal["function"]


class DatabricksMessage(TypedDict, total=False):
    role: Required[str]
    content: Required[AllDatabricksContentValues]
    tool_calls: Optional[List[DatabricksTool]]


class DatabricksChoice(TypedDict, total=False):
    index: Required[int]
    message: Required[DatabricksMessage]
    finish_reason: Required[Optional[str]]
    extra_fields: str


class DatabricksResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[DatabricksChoice]
    usage: ChatCompletionUsageBlock
