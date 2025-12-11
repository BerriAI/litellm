from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class CrowdStrikeAIDRGuardrailConfigModelOptionalParams(BaseModel):
    pass


class CrowdStrikeAIDRGuardrailConfigModel(
    GuardrailConfigModel[CrowdStrikeAIDRGuardrailConfigModelOptionalParams]
):
    api_key: Optional[str] = Field(
        default=None,
        description="The CrowdStrike AIDR API key. Reads from CS_AIDR_TOKEN env var if None.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The CrowdStrike AIDR API base URL. Reads from CS_AIDR_BASE_URL env var if None.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "CrowdStrike AIDR Guardrail"
