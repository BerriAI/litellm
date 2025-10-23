from typing import Literal, Optional

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

    guardrails_ai_api_input_format: Optional[Literal["inputs", "llmOutput"]] = Field(
        default="llmOutput",
        description="The format of the input to the Guardrails AI API. Defaults to 'llmOutput'.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Guardrails AI"
