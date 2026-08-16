from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolCallChunk
from litellm.types.utils import ChatCompletionMessageToolCall

from .base import GuardrailConfigModel

StraikerWebhookEventType = Literal["pre_call", "post_call"]
StraikerWebhookStreamPhase = Literal["none", "assembled"]
StraikerWebhookAction = Literal["NONE", "BLOCKED", "GUARDRAIL_INTERVENED"]

STRAIKER_WEBHOOK_SCHEMA_VERSION = "1"


class StraikerWebhookStream(BaseModel):
    phase: StraikerWebhookStreamPhase = "none"
    index: int | None = None


class StraikerWebhookEvent(BaseModel):
    type: StraikerWebhookEventType
    id: str
    stream: StraikerWebhookStream = Field(default_factory=StraikerWebhookStream)


class StraikerWebhookContent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    texts: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    structured_messages: list[AllMessageValues] | None = None
    tools: list[dict[str, object]] | None = None
    tool_calls: list[ChatCompletionToolCallChunk] | list[ChatCompletionMessageToolCall] | None = None
    finish_reason: str | None = None


class StraikerWebhookUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class StraikerWebhookContext(BaseModel):
    call_surface: str
    model: str | None = None
    model_provider: str | None = None
    destination: str | None = None
    session_id: str | None = None
    litellm_call_id: str | None = None
    litellm_trace_id: str | None = None
    litellm_version: str | None = None


class StraikerWebhookIdentity(BaseModel):
    litellm_key: str | None = None
    litellm_team: str | None = None
    litellm_user_id: str | None = None
    litellm_user_email: str | None = None
    litellm_org_id: str | None = None
    end_user_id: str | None = None


class StraikerWebhookApplication(BaseModel):
    source: str
    name: str | None = None


class StraikerWebhookRequest(BaseModel):
    schema_version: str = STRAIKER_WEBHOOK_SCHEMA_VERSION
    event: StraikerWebhookEvent
    request: StraikerWebhookContent
    response: StraikerWebhookContent | None = None
    context: StraikerWebhookContext
    identity: StraikerWebhookIdentity
    application: StraikerWebhookApplication
    usage: StraikerWebhookUsage | None = None
    metadata: dict[str, object] | None = None


class StraikerWebhookResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: StraikerWebhookAction = "NONE"
    blocked_reason: str | None = None
    texts: list[str] | None = None
    schema_version: str | None = None
    turn_id: str | None = Field(default=None, alias="turnId")


class StraikerGuardrailConfigModelOptionalParams(BaseModel):
    timeout: float | None = Field(
        default=5.0,
        gt=0.0,
        description="Per-attempt HTTP timeout in seconds.",
    )
    max_retries: int | None = Field(
        default=2,
        ge=0,
        description="Retries on transient HTTP (408/429/5xx) and network errors.",
    )
    initial_backoff: float | None = Field(
        default=0.1,
        ge=0.0,
        description="Initial retry backoff in seconds.",
    )
    max_backoff: float | None = Field(
        default=2.0,
        ge=0.0,
        description="Maximum retry backoff in seconds.",
    )
    unreachable_fallback: Literal["fail_open", "fail_closed"] | None = Field(
        default="fail_closed",
        description="Behavior when Straiker is unreachable after retries.",
    )
    fail_on_error: bool | None = Field(
        default=True,
        description=(
            "Behavior on any guardrail error, not just unreachability. True (default) blocks "
            "the request on error; False logs and allows the request to proceed."
        ),
    )
    max_payload_bytes: int | None = Field(
        default=524288,
        gt=0,
        description="Maximum serialized webhook payload size sent to Straiker.",
    )
    custom_headers: dict[str, str] | None = Field(
        default=None,
        description="Additional HTTP headers sent to Straiker, excluding Authorization and the webhook-format header.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description=(
            "Default metadata key/values added to the webhook metadata bag on every request. "
            "On key conflict with request-derived metadata, these configured values win."
        ),
    )
    verbose: bool | None = Field(
        default=False,
        description="Log webhook request/response payloads and record action/turn_id in response hidden params.",
    )


class StraikerGuardrailConfigModel(GuardrailConfigModel[StraikerGuardrailConfigModelOptionalParams]):
    api_key: str = Field(
        min_length=1,
        description="Straiker DefendAI environment API key (Bearer token). Env: STRAIKER_API_KEY.",
        json_schema_extra={"secret": True},
    )

    api_base: str | None = Field(
        default="https://api.prod.straiker.ai",
        description="Straiker API base URL. Use the regional variant for non-US tenants.",
    )

    default_app: str | None = Field(
        default="LiteLLM Gateway",
        description=(
            "Default application registered in the Straiker Defend Console. "
            "Overridden per-request by metadata.agent_id when present."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Straiker"
