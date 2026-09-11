from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class ThirdlawGuardrailRequestMetadata(BaseModel):
    """LiteLLM context for one guardrail evaluation, sent to the ThirdLaw intervene-service."""

    model_config = ConfigDict(frozen=True, extra="allow")

    litellm_version: str
    litellm_call_id: str | None = None
    litellm_trace_id: str | None = None
    model: str | None = None
    user_api_key_hash: str | None = None
    user_api_key_alias: str | None = None
    user_api_key_user_id: str | None = None
    user_api_key_user_email: str | None = None
    user_api_key_team_id: str | None = None
    user_api_key_team_alias: str | None = None
    user_api_key_end_user_id: str | None = None
    user_api_key_org_id: str | None = None


class ThirdlawGuardrailRequest(BaseModel):
    """POST body for the intervene-service full-payload guardrail endpoint (/guardrails/litellm/v2)."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["pre_call", "during_call", "post_call"]
    metadata: ThirdlawGuardrailRequestMetadata
    request_url: str | None = None
    request_headers: Mapping[str, str] | None = None
    request_body: Mapping[str, object] | None = None
    response_body: Mapping[str, object] | None = None
    additional_provider_specific_params: Mapping[str, object] | None = None


class ThirdlawGuardrailResponse(BaseModel):
    """Decision returned by the intervene-service.

    ``request_body`` / ``response_body`` are full replacements with shallow top-level
    overlay semantics: a top-level key that is present is authoritative and must be
    complete (e.g. a replaced ``choices`` list must carry every choice); omitted
    top-level keys keep their original values.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    action: Literal["allow", "block", "modify_request", "modify_response"]
    request_body: Mapping[str, object] | None = None
    response_body: Mapping[str, object] | None = None
    response_status: int | None = None
    message: str = ""


class ThirdlawGuardrailConfigModelOptionalParams(BaseModel):
    additional_headers: str | None = Field(
        default=None,
        description="Comma-separated list of inbound request header names whose RAW values ThirdLaw should receive. Every inbound header is always forwarded; credential headers (authorization, x-api-key, cookie) are forwarded as ***REDACTED*** unless listed here. Example: authorization,x-request-id.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description="Controls LiteLLM behavior when ThirdLaw is unreachable (network error, timeout, 502-504). fail_closed blocks the request. fail_open allows the request to continue. A block decision from ThirdLaw always blocks regardless of this setting.",
    )
    unscannable_stream_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description="Controls LiteLLM behavior when a streamed response cannot be assembled into a scannable shape (for example /v1/responses and text-completion streams). fail_closed refuses the stream. fail_open forwards it unscanned, which lets a caller pick such an endpoint to bypass response moderation.",
    )
    streaming_buffer_until_moderated: bool | None = Field(
        default=True,
        description="If true (default), a streamed response is withheld until ThirdLaw has moderated the assembled response; a modify_response decision is then applied before anything reaches the client.",
    )
    streaming_end_of_stream_only: bool | None = Field(
        default=True,
        description="If true (default), ThirdLaw is only called once, when a streamed response finishes, instead of periodically during streaming. Set to false to enable interim block-only checks via streaming_sampling_rate.",
    )
    streaming_sampling_rate: int | None = Field(
        default=5,
        description="When streaming_end_of_stream_only is false, check every Nth streamed chunk (in addition to the final end-of-stream check). Interim checks can only block, not modify. Ignored when streaming_end_of_stream_only is true.",
    )


class ThirdlawGuardrailConfigModel(GuardrailConfigModel[ThirdlawGuardrailConfigModelOptionalParams]):
    api_base: str | None = Field(
        default=None,
        description="ThirdLaw intervene-service base URL; /guardrails/litellm/v2 is appended unless already present. Env: THIRDLAW_API_BASE.",
        json_schema_extra={  # mutable-ok: pydantic stores schema extras as plain JSON containers
            "examples": [  # mutable-ok: pydantic stores schema extras as plain JSON containers
                "https://api.thirdlaw.<your-domain>",
            ]
        },
    )
    api_key: str | None = Field(
        default=None,
        description="API key for ThirdLaw, sent as a bearer token. Env: THIRDLAW_API_KEY.",
    )

    guardrail_timeout: int | None = Field(
        default=60,
        description="Timeout for the ThirdLaw API request. In seconds.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "ThirdLaw"
