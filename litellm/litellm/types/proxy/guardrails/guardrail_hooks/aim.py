from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class AimGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The API key for the Aim guardrail. If not provided, the `AIM_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The API base for the Aim guardrail. Default is https://api.aim.security. Also checks if the `AIM_API_BASE` environment variable is set.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "AIM Guardrail"
