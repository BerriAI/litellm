from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class CrowdStrikeAIDRGuardrailConfigModelOptionalParams(BaseModel):
    pass


class CrowdStrikeAIDRGuardrailConfigModel(GuardrailConfigModel[CrowdStrikeAIDRGuardrailConfigModelOptionalParams]):
    api_key: Optional[str] = Field(
        default=None,
        description="The CrowdStrike AIDR API key. Reads from CS_AIDR_TOKEN env var if None.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The CrowdStrike AIDR API base URL. Reads from CS_AIDR_BASE_URL env var if None.",
    )
    fail_on_error: Optional[bool] = Field(
        default=True,
        description="When False, transport errors from the AIDR guard API (e.g. a 4xx/5xx rejecting "
        "malformed input, as opposed to a policy block) fail open and the request proceeds unmodified.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "CrowdStrike AIDR Guardrail"
