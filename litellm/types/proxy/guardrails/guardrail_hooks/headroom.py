from typing import Literal

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class HeadroomGuardrailConfigModel(GuardrailConfigModel[BaseModel]):
    api_base: str | None = Field(
        default=None,
        description="Base URL for the headroom compression service (e.g. https://api.headroom.ai). Falls back to HEADROOM_API_BASE env var.",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the headroom compression service. Falls back to HEADROOM_API_KEY env var.",
    )
    model: str | None = Field(
        default=None,
        description="Model name forwarded to the headroom /v1/compress endpoint.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description=(
            "Behavior when the headroom compression service is unreachable or errors. "
            "'fail_closed' raises an error (default). 'fail_open' logs a critical error and "
            "forwards the request uncompressed instead of blocking it."
        ),
    )
    ccr_retrieval: bool = Field(
        default=True,
        description="Inject the Headroom retrieval tool for hashes declared by the compression service.",
    )
    min_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Skip the compression round trip when the compressible messages total fewer than this many tokens. "
            "Falls back to the HEADROOM_MIN_TOKENS env var; when neither is set, every request is compressed."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Headroom"
