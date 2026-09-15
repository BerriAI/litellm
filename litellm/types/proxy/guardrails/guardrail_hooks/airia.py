from pydantic import Field

from .base import GuardrailConfigModel


class AiriaGuardrailConfigModel(GuardrailConfigModel):
    api_base: str | None = Field(
        default=None,
        description="Base URL of the Airia AI Gateway, e.g. https://gateway.airia.ai. If not "
        "provided, the `AIRIA_GATEWAY_URL` environment variable is checked.",
    )
    api_key: str | None = Field(
        default=None,
        description="An Airia API key. If not provided, the `AIRIA_API_KEY` environment variable is checked.",
    )
    timeout: float | None = Field(
        default=None,
        description="Request timeout in seconds. If not provided, the `AIRIA_TIMEOUT` environment "
        "variable is checked, defaulting to 10.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Airia Guardrail"
