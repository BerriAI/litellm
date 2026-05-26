from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class VetoGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "Veto API key (prefix 'vt_'). If not provided, the "
            "VETO_API_KEY environment variable is used."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Veto gateway base URL. Defaults to https://api.vetocheck.com. "
            "Falls back to the VETO_API_BASE env var."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Veto"
