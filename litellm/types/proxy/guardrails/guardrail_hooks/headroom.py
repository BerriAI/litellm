from typing import Literal, Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class HeadroomGuardrailConfigModel(GuardrailConfigModel[BaseModel]):
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the headroom compression service (e.g. https://api.headroom.ai). Falls back to HEADROOM_API_BASE env var.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the headroom compression service. Falls back to HEADROOM_API_KEY env var.",
    )
    model: Optional[str] = Field(
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

    @staticmethod
    def ui_friendly_name() -> str:
        return "Headroom"
