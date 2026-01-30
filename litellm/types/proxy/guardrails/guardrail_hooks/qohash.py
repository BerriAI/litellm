from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class QohashGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The API key for the QAIGS guardrail. If not provided, the `QAIGS_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The API base URL for the QAIGS guardrail. If not provided, the `QAIGS_API_BASE` environment variable is checked. Defaults to http://qaigs:8800.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Qohash AI Guardrail Server"
