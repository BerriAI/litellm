"""
Pillar Security Guardrail Config Model
"""
from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class PillarGuardrailConfigModelOptionalParams(BaseModel):
    """Optional parameters for the Pillar Security guardrail"""

    on_flagged_action: Optional[str] = Field(
        default="monitor",
        description="Action to take when content is flagged: 'block' (raise exception) or 'monitor' (log only). If not provided, the `PILLAR_ON_FLAGGED_ACTION` environment variable is checked, defaults to 'monitor'.",
    )


class PillarGuardrailConfigModel(
    GuardrailConfigModel[PillarGuardrailConfigModelOptionalParams]
):
    """Configuration parameters for the Pillar Security guardrail"""

    api_key: Optional[str] = Field(
        default=None,
        description="API key for the Pillar Security service. If not provided, the `PILLAR_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the Pillar Security API. If not provided, the `PILLAR_API_BASE` environment variable is checked, defaults to https://api.pillar.security",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Pillar Guardrail"
