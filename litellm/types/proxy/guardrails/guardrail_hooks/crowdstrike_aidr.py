from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class CrowdStrikeAIDRGuardrailConfigModelOptionalParams(BaseModel):
    streaming_buffer_until_moderated: bool | None = Field(
        default=None,
        description="When True, withhold streamed chunks until moderation passes. Defaults to False when unset.",
    )
    streaming_buffer_release_on_scan: bool | None = Field(
        default=None,
        description="When buffering, release withheld chunks after each passing scan. Defaults to False when unset.",
    )
    streaming_end_of_stream_only: bool | None = Field(
        default=None,
        description="If True (default when unset), the guard runs once over the assembled response at end of "
        "stream, so flagged content may already have reached the client. If False, post_call scans the "
        "accumulated streamed response every streaming_sampling_rate chunks and an in-flight block stops the stream.",
    )
    streaming_sampling_rate: int | None = Field(
        default=None,
        ge=1,
        description="When streaming_end_of_stream_only is False, scan the accumulated streamed response every Nth "
        "chunk. Defaults to 5 when unset.",
    )


class CrowdStrikeAIDRGuardrailConfigModel(GuardrailConfigModel[CrowdStrikeAIDRGuardrailConfigModelOptionalParams]):
    api_key: str | None = Field(
        default=None,
        description="The CrowdStrike AIDR API key. Reads from CS_AIDR_TOKEN env var if None.",
    )
    api_base: str | None = Field(
        default=None,
        description="The CrowdStrike AIDR API base URL. Reads from CS_AIDR_BASE_URL env var if None.",
    )
    fail_on_error: bool | None = Field(
        default=True,
        description="When False, errors calling the AIDR guard API (connection failures, timeouts, 4xx/5xx "
        "responses, malformed reply bodies) fail open and the request proceeds unmodified. A blocked verdict "
        "delivered on a success response still blocks, and a transformed response that cannot be parsed "
        "fails closed so delivered redactions are never dropped.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "CrowdStrike AIDR Guardrail"
