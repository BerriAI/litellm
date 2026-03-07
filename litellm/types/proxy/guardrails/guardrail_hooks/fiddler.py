from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class FiddlerGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The API key for the Fiddler guardrail. If not provided, the `FIDDLER_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The base URL for the Fiddler guardrail API. Defaults to https://app.fiddler.ai. Also checks the `FIDDLER_API_BASE` environment variable.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Fiddler Guardrail"
