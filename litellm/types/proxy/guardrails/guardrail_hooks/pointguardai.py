from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class PointGuardAIGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the PointGuardAI v2 guardrail"""

    org_code: Optional[str] = Field(
        default=None,
        description="Organization ID for PointGuardAI",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Base API for the PointGuardAI service",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API KEY for the PointGuardAI service",
    )
    policy_config_name: Optional[str] = Field(
        default=None,
        description="Policy configuration name for PointGuardAI",
    )
    correlation_key: Optional[str] = Field(
        default=None,
        description="Optional correlation key for request tracking",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "PointGuard AI"
