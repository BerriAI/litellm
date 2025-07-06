from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class GuardrailsAIGuardrailConfigModel(GuardrailConfigModel):
    api_base: Optional[str] = Field(
        default=None,
        description="The API base for the Guardrails AI guardrail. Defaults to http://0.0.0.0:8000, the `GUARDRAILS_AI_API_BASE` environment variable is checked.",
    )

    guard_name: Optional[str] = Field(
        default=None,
        description="The name of the Guardrails AI guardrail. Required for the Guardrails AI guardrail.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Guardrails AI"
