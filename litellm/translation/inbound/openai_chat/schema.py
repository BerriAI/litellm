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

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Sampling = int | float


class WireModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class CacheControlIn(WireModel):
    type: Literal["ephemeral"]
    ttl: str | None = None


class TextPartIn(WireModel):
    type: Literal["text"]
    text: str
    cache_control: CacheControlIn | None = None


class ImageUrlIn(WireModel):
    url: str
    format: str | None = None
    detail: str | None = None


class ImagePartIn(WireModel):
    type: Literal["image_url"]
    image_url: str | ImageUrlIn
    cache_control: CacheControlIn | None = None


UserPartIn = Annotated[TextPartIn | ImagePartIn, Field(discriminator="type")]


class ThinkingPartIn(WireModel):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None
    cache_control: CacheControlIn | None = None


class RedactedThinkingPartIn(WireModel):
    type: Literal["redacted_thinking"]
    data: str


AssistantPartIn = Annotated[
    TextPartIn | ThinkingPartIn | RedactedThinkingPartIn,
    Field(discriminator="type"),
]

ThinkingBlockIn = Annotated[
    ThinkingPartIn | RedactedThinkingPartIn, Field(discriminator="type")
]


class ToolCallFunctionIn(WireModel):
    name: str
    arguments: str | None = None


class ToolCallIn(WireModel):
    id: str
    type: str
    function: ToolCallFunctionIn | None = None
    index: int | None = None
    cache_control: CacheControlIn | None = None


class SystemMessageIn(WireModel):
    role: Literal["system"]
    content: str | list[TextPartIn] | None = None
    name: str | None = None
    cache_control: CacheControlIn | None = None


class UserMessageIn(WireModel):
    role: Literal["user"]
    content: str | list[UserPartIn] | None = None
    name: str | None = None
    cache_control: CacheControlIn | None = None


class AssistantMessageIn(WireModel):
    role: Literal["assistant"]
    content: str | list[AssistantPartIn] | None = None
    tool_calls: list[ToolCallIn] | None = None
    thinking_blocks: list[ThinkingBlockIn] | None = None
    name: str | None = None
    cache_control: CacheControlIn | None = None
    refusal: object | None = None
    annotations: object | None = None
    audio: object | None = None
    function_call: object | None = None
    provider_specific_fields: object | None = None


class ToolMessageIn(WireModel):
    role: Literal["tool"]
    tool_call_id: str
    content: str | list[TextPartIn] | None = None
    name: str | None = None
    cache_control: CacheControlIn | None = None


MessageIn = Annotated[
    SystemMessageIn | UserMessageIn | AssistantMessageIn | ToolMessageIn,
    Field(discriminator="role"),
]


class ToolFunctionIn(WireModel):
    name: str
    description: str | None = None
    parameters: object | None = None
    strict: bool | None = None
    cache_control: CacheControlIn | None = None
    defer_loading: object | None = None
    allowed_callers: object | None = None
    input_examples: object | None = None


class ToolIn(WireModel):
    type: Literal["function", "custom"]
    function: ToolFunctionIn
    cache_control: CacheControlIn | None = None
    defer_loading: object | None = None
    allowed_callers: object | None = None
    input_examples: object | None = None


class ToolChoiceFunctionIn(WireModel):
    name: str


class ToolChoiceNamedIn(WireModel):
    type: str | None = None
    function: ToolChoiceFunctionIn


class ToolChoiceTypeOnlyIn(WireModel):
    type: Literal["auto", "required", "any", "none"]


ToolChoiceIn = (
    Literal["auto", "required", "none"] | ToolChoiceNamedIn | ToolChoiceTypeOnlyIn
)


class JsonSchemaIn(WireModel):
    json_schema: object = Field(alias="schema")
    name: str | None = None
    description: str | None = None
    strict: bool | None = None


class ResponseFormatIn(WireModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaIn | None = None


class ReasoningEffortObjectIn(WireModel):
    """Dict form sent by Responses-bridge callers: only ``effort`` is read by
    v1; ``summary`` is accepted there and ignored."""

    effort: Literal["minimal", "low", "medium", "high", "xhigh", "max", "none"]
    summary: object | None = None


ReasoningEffortIn = (
    Literal["minimal", "low", "medium", "high", "xhigh", "max", "none"]
    | ReasoningEffortObjectIn
)


class ThinkingIn(WireModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: int | None = None


class ChatRequestIn(WireModel):
    model: str
    messages: list[MessageIn]
    stream: bool | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: Sampling | None = None
    top_p: Sampling | None = None
    top_k: Sampling | None = None
    stop: str | list[str] | None = None
    tools: list[ToolIn] | None = None
    tool_choice: ToolChoiceIn | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None
    response_format: ResponseFormatIn | None = None
    reasoning_effort: ReasoningEffortIn | None = None
    thinking: ThinkingIn | None = None
