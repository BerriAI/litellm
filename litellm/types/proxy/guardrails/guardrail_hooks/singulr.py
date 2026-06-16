from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class SingulrGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for Singulr authentication. "
            "If not provided, the SINGULR_API_KEY "
            "environment variable is used."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Singulr Guardrails API base URL. "
            "Falls back to SINGULR_API_BASE env var."
        ),
    )
    enforcement_entity_id: Optional[str] = Field(
        default=None,
        description=(
            "The enforcement entity ID (e.g., Application ID or Agent ID) "
            "to send in the X-Singulr-Enforcement-Entity-Id header."
        ),
    )
    sdk_guardrail_id: Optional[str] = Field(
        default=None,
        description=(
            "The SDK guardrail ID to send in the X-Singulr-Guardrail-Id header."
        ),
    )
    block_on_error: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to block the request when the "
            "Singulr API is unreachable or returns an error. "
            "Defaults to true (fail-closed)."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Singulr"
