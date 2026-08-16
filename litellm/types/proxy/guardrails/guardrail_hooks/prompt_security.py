from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class PromptSecurityGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The API key for the Prompt Security guardrail. If not provided, the `PROMPT_SECURITY_API_KEY` environment variable is used.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The API base for the Prompt Security guardrail. If not provided, the `PROMPT_SECURITY_API_BASE` environment variable is used.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Prompt Security"
