from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class AporiaGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The API key for the Aporia guardrail. If not provided, the `APORIA_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The API base for the Aporia guardrail. If not provided, the `APORIA_API_BASE` environment variable is checked.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Aporia AI"
