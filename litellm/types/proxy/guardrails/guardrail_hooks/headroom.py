from typing import Optional

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

    @staticmethod
    def ui_friendly_name() -> str:
        return "Headroom"
