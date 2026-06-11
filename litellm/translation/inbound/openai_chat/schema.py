"""Frozen pydantic v2 wire models for the OpenAI chat-completions request.

This is the fail-closed allowlist: every model is ``extra="forbid"`` and
``strict=True``, so a field the schema does not name is a typed error (the
seam falls back to v1 on it), never a silent drop. Fields v1 demonstrably
ignores (``name`` on messages, ``index`` on replayed tool calls, ``detail``
on image URLs, assistant ``refusal``/``annotations``/``audio``) are accepted
here and deliberately not carried into the IR, which is the same observable
behavior as v1. Fields v1 acts on through paths v2 has not ported (legacy
``function_call``, ``provider_specific_fields`` server-tool payloads) are
accepted as ``object`` and rejected with ``unsupported`` in ``parse`` when
they carry a value, so they fall back to v1 instead of being lost.

Numbers use ``Union[int, float]`` plus strict mode so an int stays an int on
the wire (v1 forwards caller bytes; ``temperature: 1`` must not become 1.0).
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

Sampling = Union[int, float]


class WireModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class CacheControlIn(WireModel):
    type: Literal["ephemeral"]
    ttl: Optional[str] = None


class TextPartIn(WireModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[CacheControlIn] = None


class ImageUrlIn(WireModel):
    url: str
    format: Optional[str] = None
    detail: Optional[str] = None


class ImagePartIn(WireModel):
    type: Literal["image_url"]
    image_url: Union[str, ImageUrlIn]
    cache_control: Optional[CacheControlIn] = None


UserPartIn = Annotated[Union[TextPartIn, ImagePartIn], Field(discriminator="type")]


class ThinkingPartIn(WireModel):
    type: Literal["thinking"]
    thinking: str
    signature: Optional[str] = None
    cache_control: Optional[CacheControlIn] = None


class RedactedThinkingPartIn(WireModel):
    type: Literal["redacted_thinking"]
    data: str


AssistantPartIn = Annotated[
    Union[TextPartIn, ThinkingPartIn, RedactedThinkingPartIn],
    Field(discriminator="type"),
]

ThinkingBlockIn = Annotated[Union[ThinkingPartIn, RedactedThinkingPartIn], Field(discriminator="type")]


class ToolCallFunctionIn(WireModel):
    name: str
    arguments: Optional[str] = None


class ToolCallIn(WireModel):
    id: str
    type: str
    function: Optional[ToolCallFunctionIn] = None
    index: Optional[int] = None
    cache_control: Optional[CacheControlIn] = None


class SystemMessageIn(WireModel):
    role: Literal["system"]
    content: Union[str, List[TextPartIn], None] = None
    name: Optional[str] = None
    cache_control: Optional[CacheControlIn] = None


class UserMessageIn(WireModel):
    role: Literal["user"]
    content: Union[str, List[UserPartIn], None] = None
    name: Optional[str] = None
    cache_control: Optional[CacheControlIn] = None


class AssistantMessageIn(WireModel):
    role: Literal["assistant"]
    content: Union[str, List[AssistantPartIn], None] = None
    tool_calls: Optional[List[ToolCallIn]] = None
    thinking_blocks: Optional[List[ThinkingBlockIn]] = None
    name: Optional[str] = None
    cache_control: Optional[CacheControlIn] = None
    refusal: Optional[object] = None
    annotations: Optional[object] = None
    audio: Optional[object] = None
    function_call: Optional[object] = None
    provider_specific_fields: Optional[object] = None


class ToolMessageIn(WireModel):
    role: Literal["tool"]
    tool_call_id: str
    content: Union[str, List[TextPartIn], None] = None
    name: Optional[str] = None
    cache_control: Optional[CacheControlIn] = None


MessageIn = Annotated[
    Union[SystemMessageIn, UserMessageIn, AssistantMessageIn, ToolMessageIn],
    Field(discriminator="role"),
]


class ToolFunctionIn(WireModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[object] = None
    strict: Optional[bool] = None
    cache_control: Optional[CacheControlIn] = None
    defer_loading: Optional[object] = None
    allowed_callers: Optional[object] = None
    input_examples: Optional[object] = None


class ToolIn(WireModel):
    type: Literal["function", "custom"]
    function: ToolFunctionIn
    cache_control: Optional[CacheControlIn] = None
    defer_loading: Optional[object] = None
    allowed_callers: Optional[object] = None
    input_examples: Optional[object] = None


class ToolChoiceFunctionIn(WireModel):
    name: str


class ToolChoiceNamedIn(WireModel):
    type: Optional[str] = None
    function: ToolChoiceFunctionIn


class ToolChoiceTypeOnlyIn(WireModel):
    type: Literal["auto", "required", "any", "none"]


ToolChoiceIn = Union[Literal["auto", "required", "none"], ToolChoiceNamedIn, ToolChoiceTypeOnlyIn]


class JsonSchemaIn(WireModel):
    json_schema: object = Field(alias="schema")
    name: Optional[str] = None
    description: Optional[str] = None
    strict: Optional[bool] = None


class ResponseFormatIn(WireModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaIn] = None


class ReasoningEffortObjectIn(WireModel):
    """Dict form sent by Responses-bridge callers: only ``effort`` is read by
    v1; ``summary`` is accepted there and ignored."""

    effort: Literal["minimal", "low", "medium", "high", "xhigh", "max", "none"]
    summary: Optional[object] = None


ReasoningEffortIn = Union[
    Literal["minimal", "low", "medium", "high", "xhigh", "max", "none"],
    ReasoningEffortObjectIn,
]


class ThinkingIn(WireModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None


class ChatRequestIn(WireModel):
    model: str
    messages: List[MessageIn]
    stream: Optional[bool] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[Sampling] = None
    top_p: Optional[Sampling] = None
    top_k: Optional[Sampling] = None
    stop: Union[str, List[str], None] = None
    tools: Optional[List[ToolIn]] = None
    tool_choice: Optional[ToolChoiceIn] = None
    parallel_tool_calls: Optional[bool] = None
    user: Optional[str] = None
    response_format: Optional[ResponseFormatIn] = None
    reasoning_effort: Optional[ReasoningEffortIn] = None
    thinking: Optional[ThinkingIn] = None
