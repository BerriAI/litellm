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
    async_mode: Optional[bool] = Field(
        default=None,
        description="Set to True to request asynchronous analysis (sets `plr_async` header).",
    )
    persist_session: Optional[bool] = Field(
        default=None,
        description="Set to False to disable session persistence (sets `plr_persist` header).",
    )
    include_scanners: Optional[bool] = Field(
        default=True,
        description="Include scanner summaries in response payloads (sets `plr_scanners` header).",
    )
    include_evidence: Optional[bool] = Field(
        default=True,
        description="Include detailed evidence objects in response payloads (sets `plr_evidence` header).",
    )
    fallback_on_error: Optional[str] = Field(
        default=None,
        description="Action to take when Pillar API is unavailable or errors: 'allow' (proceed without scanning) or 'block' (reject request with 503 error). If not provided, the `PILLAR_FALLBACK_ON_ERROR` environment variable is checked, defaults to 'allow'.",
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Timeout in seconds for Pillar API calls. If not provided, the `PILLAR_TIMEOUT` environment variable is checked, defaults to 5.0 seconds.",
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
