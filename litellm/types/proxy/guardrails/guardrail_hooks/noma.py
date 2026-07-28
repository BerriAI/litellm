from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class NomaGuardrailConfigModel(GuardrailConfigModel):
    use_v2: Optional[bool] = Field(
        default=False,
        description="If True and guardrail='noma', route to the new Noma v2 implementation.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The Noma API key. Reads from NOMA_API_KEY env var if None.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The Noma API base URL. Defaults to https://api.noma.security. Also checks if the NOMA_API_KEY env var is set.",
    )
    application_id: Optional[str] = Field(
        default=None,
        description="The Noma Application ID. Reads from NOMA_APPLICATION_ID env var if None.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Noma Security"


class NomaV2GuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The Noma API key. Reads from NOMA_API_KEY env var if None.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The Noma API base URL. Defaults to https://api.noma.security.",
    )
    application_id: Optional[str] = Field(
        default=None,
        description="The Noma Application ID. Reads from NOMA_APPLICATION_ID env var if None.",
    )
    monitor_mode: Optional[bool] = Field(
        default=None,
        description="When true, run guardrail checks in monitor mode.",
    )
    block_failures: Optional[bool] = Field(
        default=None,
        description="When true, fail closed on Noma API errors.",
    )
    streaming_end_of_stream_only: Optional[bool] = Field(
        default=None,
        description=(
            "If False (default when unset), streaming post-call scans run on sampled chunks at the cadence set "
            "by streaming_sampling_rate, and an in-flight block stops further chunks from streaming. If True, the "
            "scan runs once at end of stream over the assembled response; lower cost and latency, but flagged "
            "content has already streamed to the client before the terminal block."
        ),
    )
    streaming_sampling_rate: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "When streaming_end_of_stream_only is False, the streaming post-call scan runs every Nth streamed "
            "chunk. Ignored when streaming_end_of_stream_only is True. Must be >= 1 when set; defaults to 5."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Noma Security v2"
