"""Frozen pydantic v2 wire models for the Anthropic Messages request.

The fail-closed allowlist (every model ``extra="forbid"`` and ``strict=True``):
a field the schema does not name is a typed error and the seam falls back to
v1, never a silent drop. The divergence from the OpenAI chat schema this
package mirrors: ``max_tokens`` is REQUIRED (Anthropic rejects a request
without it; v1's adapter raises), ``system`` is ``str | list[block]``, stop
sequences ride the ``stop_sequences`` key, and ``tool_choice``/``thinking`` are
the Anthropic unions.

Fields v1 serves through paths the chat IR cannot carry (documents, hosted
tools, ``output_format``/``output_config``, ``mcp_servers``, ``container``,
metadata beyond ``user_id``, ``thinking.summary``, version/beta envelope) are
accepted as ``object`` here and rejected with ``unsupported`` in ``parse`` when
present, so they fall back to v1 instead of being lost (researcher-6 §1.4/§1.5).
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


class TextBlockIn(WireModel):
    type: Literal["text"]
    text: str
    cache_control: CacheControlIn | None = None


class Base64SourceIn(WireModel):
    type: Literal["base64"]
    media_type: str
    data: str


class UrlSourceIn(WireModel):
    type: Literal["url"]
    url: str


class ImageBlockIn(WireModel):
    type: Literal["image"]
    source: Base64SourceIn | UrlSourceIn | object
    cache_control: CacheControlIn | None = None


class ToolResultContentTextIn(WireModel):
    type: Literal["text"]
    text: str
    cache_control: CacheControlIn | None = None


class ToolResultBlockIn(WireModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[ToolResultContentTextIn | object] | None = None
    is_error: bool | None = None
    cache_control: CacheControlIn | None = None


UserBlockIn = Annotated[
    TextBlockIn | ImageBlockIn | ToolResultBlockIn,
    Field(discriminator="type"),
]


class ToolUseBlockIn(WireModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: object
    cache_control: CacheControlIn | None = None
    caller: object | None = None


class ThinkingBlockIn(WireModel):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None
    cache_control: CacheControlIn | None = None


class RedactedThinkingBlockIn(WireModel):
    type: Literal["redacted_thinking"]
    data: str


AssistantBlockIn = Annotated[
    TextBlockIn | ToolUseBlockIn | ThinkingBlockIn | RedactedThinkingBlockIn,
    Field(discriminator="type"),
]


class UserMessageIn(WireModel):
    role: Literal["user"]
    content: str | list[UserBlockIn]


class AssistantMessageIn(WireModel):
    role: Literal["assistant"]
    content: str | list[AssistantBlockIn]
    name: str | None = None


MessageIn = Annotated[
    UserMessageIn | AssistantMessageIn,
    Field(discriminator="role"),
]


class SystemTextBlockIn(WireModel):
    type: str
    text: str | None = None
    cache_control: CacheControlIn | None = None


class ToolIn(WireModel):
    name: str
    description: str | None = None
    input_schema: object | None = None
    type: Literal["custom"] | None = None
    cache_control: CacheControlIn | None = None
    defer_loading: object | None = None
    allowed_callers: object | None = None
    input_examples: object | None = None


class ToolChoiceIn(WireModel):
    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None
    disable_parallel_tool_use: bool | None = None


class ThinkingIn(WireModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: int | None = None
    summary: object | None = None


class MetadataIn(WireModel):
    user_id: str | None = None


class AnthropicMessagesRequestIn(WireModel):
    model: str
    messages: list[MessageIn]
    max_tokens: int
    system: str | list[SystemTextBlockIn] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = None
    temperature: Sampling | None = None
    top_p: Sampling | None = None
    top_k: Sampling | None = None
    tools: list[ToolIn] | None = None
    tool_choice: ToolChoiceIn | None = None
    thinking: ThinkingIn | None = None
    metadata: MetadataIn | None = None
    # Accepted-as-object then rejected in parse when present (each names its
    # v1 path); the chat IR has no home for any of these (researcher-6 §1.5).
    output_format: object | None = None
    output_config: object | None = None
    mcp_servers: object | None = None
    container: object | None = None
    context_management: object | None = None
    reasoning_effort: object | None = None
    inference_geo: object | None = None
    speed: object | None = None
    cache_control: object | None = None
    service_tier: object | None = None
