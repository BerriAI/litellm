from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class CatoNetworksGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The API key for the Cato Networks guardrail. If not provided, the `CATO_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The API base for the Cato Networks guardrail. Default is https://api.aisec.catonetworks.com. Also checks if the `CATO_API_BASE` environment variable is set.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Cato Networks Guardrail"
