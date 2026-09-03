from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class ContentBlock(BaseModel):
    type: str | None = None
    text: str | None = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | Sequence[ContentBlock] | None = None
    tool_calls: Sequence[ToolCall] | None = None


class SingulrGuardrailPayload(BaseModel):
    correlation_id: str | None = None
    model_name: str | None = None
    model_provider_name: str | None = None
    guardrail_scope: str | None = None
    messages: Sequence[Any] | None = None
    images: Sequence[str] | None = None
    tools: Sequence[Any] | None = None  # pyright: ignore[reportExplicitAny]  # forwards caller-supplied OpenAI tool defs verbatim
    response: Any = None  # pyright: ignore[reportExplicitAny]  # logging_only reports raw litellm callback results (ModelResponse, EmbeddingResponse, etc.)
    metadata: Mapping[str, Any] | None = None


class SingulrMcpGuardrailPayload(BaseModel):
    model_name: str | None = None
    guardrail_scope: str | None = None
    tool_name: str | None = None
    tool_arguments: Mapping[str, Any] | None = None
    mcp_server_name: str | None = None
    tool_result: Sequence[str] | None = None
    metadata: Mapping[str, Any] | None = None


class SingulrGuardrailResponse(BaseModel):
    """Response returned by the Singulr guardrail API."""

    should_block: bool | None = None
    blocking_due_to: str | None = None


class SingulrGuardrailConfigModel(GuardrailConfigModel):
    singulr_api_key: str | None = Field(
        default=None,
        description="The Singulr API key. Generate API key from Singulr Platform.",
    )

    singulr_api_base: str | None = Field(
        default=None,
        description="The Singulr API base URL. Get base URL from Singulr Platform.",
    )

    singulr_application_id: str | None = Field(
        default=None,
        description="The Singulr application ID. Get application ID from Singulr Platform.",
    )

    singulr_guardrail_id: str | None = Field(
        default=None,
        description="The Singulr Guardrail ID. Get guardrail ID from Singulr Platform.",
    )

    block_on_error: bool | None = Field(
        default=None,
        description=(
            "Whether to block requests when the Singulr Guardrails API is unavailable "
            "or returns an error. If enabled, requests fail closed. "
            "If disabled, requests continue without guardrail enforcement (fail open)."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Singulr"
