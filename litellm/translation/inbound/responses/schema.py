"""Frozen pydantic v2 wire models for the OpenAI Responses request.

The fail-closed allowlist (every model ``extra="forbid"`` and ``strict=True``):
a field the schema does not name is a typed error and the seam falls back to
v1, never a silent drop. The Responses surface is the v1 supported-param
allowlist (``get_supported_openai_params``): ``input``, ``instructions``,
``max_output_tokens``, ``parallel_tool_calls``, ``reasoning``, ``stream``,
``temperature``, ``text``, ``tool_choice``, ``tools``, ``top_p``, ``user``,
plus the fields v1 reads but the chat IR cannot carry (``previous_response_id``,
``metadata``, ``service_tier``, ``include``, ``store``, ``background``), which
are accepted here and rejected with ``unsupported`` in ``parse`` so they fall
back to v1 instead of being lost (researcher-6 §2.4/§2.5).

The ``input`` list is a polymorphic item union: a message (``type`` absent or
``"message"``), a ``function_call``, a ``function_call_output``/``tool_result``,
a ``reasoning`` item, an ``item_reference``, and the hosted-tool call/output
items. Message and function_call/function_call_output are served; the rest fail
closed in ``parse`` (they need session state or hosted-tool semantics the IR
has no home for).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Sampling = int | float


class WireModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class InputTextContentIn(WireModel):
    type: Literal["input_text", "output_text", "text"]
    text: str | None = None


class ImageUrlContentIn(WireModel):
    type: Literal["input_image"]
    image_url: str | None = None
    detail: str | None = None
    file_id: str | None = None


class InputFileContentIn(WireModel):
    type: Literal["input_file"]
    file_id: str | None = None
    file_url: str | None = None
    file_data: str | None = None
    cache_control: object | None = None


MessageContentPartIn = Annotated[
    InputTextContentIn | ImageUrlContentIn | InputFileContentIn,
    Field(discriminator="type"),
]


class MessageItemIn(WireModel):
    type: Literal["message"] | None = None
    role: str
    content: str | list[MessageContentPartIn]
    status: str | None = None


class FunctionCallItemIn(WireModel):
    type: Literal["function_call"]
    call_id: str | None = None
    id: str | None = None
    name: str | None = None
    arguments: str | None = None
    status: str | None = None


class FunctionCallOutputItemIn(WireModel):
    type: Literal["function_call_output", "tool_result"]
    call_id: str | None = None
    output: str | None = None
    status: str | None = None


class ReasoningItemIn(WireModel):
    type: Literal["reasoning"]
    id: str | None = None
    summary: object | None = None
    content: object | None = None
    encrypted_content: object | None = None


class ItemReferenceIn(WireModel):
    type: Literal["item_reference"]
    id: str | None = None


InputItemIn = (
    MessageItemIn
    | FunctionCallItemIn
    | FunctionCallOutputItemIn
    | ReasoningItemIn
    | ItemReferenceIn
)


class FunctionToolIn(WireModel):
    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: object | None = None
    strict: bool | None = None
    cache_control: object | None = None
    defer_loading: object | None = None
    allowed_callers: object | None = None
    input_examples: object | None = None


class TextFormatIn(WireModel):
    type: Literal["text", "json_object", "json_schema"]
    name: str | None = None
    schema_: object | None = Field(default=None, alias="schema")
    strict: bool | None = None
    description: str | None = None


class TextIn(WireModel):
    format: TextFormatIn | None = None


class ReasoningIn(WireModel):
    effort: (
        Literal["minimal", "low", "medium", "high", "xhigh", "max", "none"] | None
    ) = None
    summary: object | None = None
    generate_summary: object | None = None


class ToolChoiceNamedIn(WireModel):
    type: Literal["function", "tool", "any", "auto", "none", "required"]
    name: str | None = None


class ResponsesRequestIn(WireModel):
    model: str
    input: str | list[InputItemIn]
    instructions: str | None = None
    max_output_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    reasoning: ReasoningIn | None = None
    stream: bool | None = None
    temperature: Sampling | None = None
    text: TextIn | None = None
    tool_choice: str | ToolChoiceNamedIn | None = None
    tools: list[FunctionToolIn | object] | None = None
    top_p: Sampling | None = None
    user: str | None = None
    # Accepted-as-object then rejected in parse when present (each names its
    # v1 path); the chat IR has no home for any of these (researcher-6 §2.5).
    previous_response_id: object | None = None
    metadata: object | None = None
    service_tier: object | None = None
    include: object | None = None
    store: object | None = None
    background: object | None = None
    truncation: object | None = None
