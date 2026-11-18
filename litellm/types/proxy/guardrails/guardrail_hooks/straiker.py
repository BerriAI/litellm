from __future__ import annotations

from typing import Final, Literal, TypeAlias

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel

from .base import GuardrailConfigModel

StraikerWebhookEventType = Literal["pre_call", "post_call"]
StraikerWebhookStreamPhase = Literal["none", "assembled"]
StraikerWebhookAction = Literal["NONE", "BLOCKED", "GUARDRAIL_INTERVENED"]
StraikerStreamMode: TypeAlias = Literal["buffered", "streaming"]

STRAIKER_WEBHOOK_SCHEMA_VERSION: Final = "1"


class StraikerWebhookStream(LiteLLMBaseModel):
    phase: StraikerWebhookStreamPhase = "none"
    index: int | None = None


class StraikerWebhookEvent(LiteLLMBaseModel):
    type: StraikerWebhookEventType
    id: str
    stream: StraikerWebhookStream = Field(default_factory=StraikerWebhookStream)


class StraikerWebhookContent(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore")

    texts: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    structured_messages: list[dict[str, object]] | None = None
    tools: list[dict[str, object]] | None = None
    tool_calls: list[dict[str, object]] | None = None
    finish_reason: str | None = None


class StraikerWebhookUsage(LiteLLMBaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class StraikerWebhookContext(LiteLLMBaseModel):
    call_surface: str
    mode: list[str] | None = None
    model: str | None = None
    model_provider: str | None = None
    destination: str | None = None
    session_id: str | None = None
    litellm_call_id: str | None = None
    litellm_trace_id: str | None = None
    litellm_version: str | None = None


class StraikerWebhookIdentity(LiteLLMBaseModel):
    litellm_key: str | None = None
    litellm_team: str | None = None
    litellm_user_id: str | None = None
    litellm_user_email: str | None = None
    litellm_org_id: str | None = None
    end_user_id: str | None = None


class StraikerWebhookApplication(LiteLLMBaseModel):
    source: str
    name: str | None = None


class StraikerWebhookRequest(LiteLLMBaseModel):
    schema_version: str = STRAIKER_WEBHOOK_SCHEMA_VERSION
    event: StraikerWebhookEvent
    request: StraikerWebhookContent
    response: StraikerWebhookContent | None = None
    context: StraikerWebhookContext
    identity: StraikerWebhookIdentity
    application: StraikerWebhookApplication
    usage: StraikerWebhookUsage | None = None
    metadata: dict[str, object] | None = None


class StraikerWebhookResponse(LiteLLMBaseModel):
    model_config = ConfigDict(extra="allow")

    action: StraikerWebhookAction = "NONE"
    blocked_reason: str | None = None
    #: The controls that blocked this turn, when the platform names them. Empty for a block
    #: that comes from state rather than content, such as an engaged kill switch.
    blocked_by: tuple[str, ...] = ()
    texts: list[str] | None = None
    schema_version: str | None = None
    turn_id: str | None = Field(default=None, alias="turnId")
    #: The v3 reply's own answer, `allow`, `detect` or `block`, and the turn's details, as the
    #: guardrail logs them. `action` above is the v1 vocabulary the guardrail acts on.
    verdict: str | None = None
    #: Every control that fired, in detect or block mode; `blocked_by` is the subset that blocked.
    controls: tuple[str, ...] = ()
    events_scored: int | None = None
    session_id: str | None = None


class StraikerGuardrailConfigModelOptionalParams(LiteLLMBaseModel):
    timeout: float | None = Field(
        default=5.0,
        gt=0.0,
        description="Per-attempt HTTP timeout in seconds. An attempt that times out is not retried.",
    )
    max_retries: int | None = Field(
        default=2,
        ge=0,
        description=(
            "Retries on HTTP 408, 429, 500, 502, 503 and 504 and on connection errors. Timeouts are not retried."
        ),
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
        description=(
            "Behavior when Straiker cannot give a verdict: a timeout, a connection error, or HTTP "
            "408, 429, 500, 502, 503 or 504 once retries are spent. Other errors follow fail_on_error."
        ),
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
    stream_mode: StraikerStreamMode | None = Field(
        default="buffered",
        description=(
            "How a streamed response reaches the client when post_call is on. 'buffered' (default) "
            "withholds every chunk until Straiker has checked the whole response, so a blocked "
            "response never reaches the client. 'streaming' sends chunks as they arrive and checks "
            "the assembled response at the end, holding back only the stream's final frames: a "
            "block then ends the stream with the block message, but the text before it has "
            "already been delivered."
        ),
    )
    api_version: Literal["v1", "v3"] | None = Field(
        default=None,
        description=(
            "Straiker detect API the gateway calls. 'v1' posts the structured webhook envelope "
            "to /api/v1/detect/webhook (legacy Defend, UUID collection key). 'v3' relays the "
            "provider request and response to /api/v3/detect, the v3 platform's only detect "
            "route, which accepts only an sk_agt_ integration key. Unset: chosen from the key "
            "prefix, so a v3 key needs no extra configuration."
        ),
    )
    agent_ref: str | None = Field(
        default=None,
        description=(
            "v3 only. Names the Straiker agent this route's traffic belongs to when one gateway "
            "fronts several applications, sent as x-s6r-agent. It wins over a client-supplied "
            "x-s6r-agent header. Names ONE agent, never a kind of agent: Straiker keys per-agent state on it, so "
            "sharing a value across applications merges them into one agent."
        ),
    )
    client: str | None = Field(
        default=None,
        description=(
            "v3 only. Optional x-s6r-client routing hint. Leave unset on a shared gateway; set it on a "
            "route that serves a single application."
        ),
    )
    format_hint: Literal["anthropic.messages", "openai.chat"] | None = Field(
        default=None,
        description="v3 only. Optional x-s6r-format hint. Only breaks the messages-array tie between formats.",
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
