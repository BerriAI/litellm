from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class VigilGuardGuardrailConfigModel(GuardrailConfigModel):
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Vigil Guard API base URL. "
            "Falls back to the VIGIL_GUARD_URL environment variable."
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "Vigil Guard API key. "
            "Falls back to the VIGIL_GUARD_API_KEY environment variable."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Vigil Guard"
