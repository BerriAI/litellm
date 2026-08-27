from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class CrowdStrikeAIDRGuardrailConfigModelOptionalParams(BaseModel):
    pass


class CrowdStrikeAIDRGuardrailConfigModel(GuardrailConfigModel[CrowdStrikeAIDRGuardrailConfigModelOptionalParams]):
    api_key: str | None = Field(
        default=None,
        description="The CrowdStrike AIDR API key. Reads from CS_AIDR_TOKEN env var if None.",
    )
    api_base: str | None = Field(
        default=None,
        description="The CrowdStrike AIDR API base URL. Reads from CS_AIDR_BASE_URL env var if None.",
    )
    fail_on_error: bool | None = Field(
        default=True,
        description="When False, errors calling the AIDR guard API (connection failures, timeouts, 4xx/5xx "
        "responses, malformed reply bodies) fail open and the request proceeds unmodified. A blocked verdict "
        "delivered on a success response still blocks, and a transformed response that cannot be parsed "
        "fails closed so delivered redactions are never dropped.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "CrowdStrike AIDR Guardrail"
