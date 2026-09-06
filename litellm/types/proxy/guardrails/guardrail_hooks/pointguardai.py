from typing import Literal

from pydantic import Field

from .base import GuardrailConfigModel


class PointGuardAIGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the PointGuardAI v2 guardrail"""

    org_code: str | None = Field(
        default=None,
        description="Organization code for PointGuardAI.",
    )
    api_base: str | None = Field(
        default=None,
        description="Base URL for PointGuardAI. Defaults to https://api.appsoc.com.",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for PointGuardAI.",
    )
    policy_config_name: str | None = Field(
        default=None,
        description="PointGuardAI policy configuration name.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description=(
            "Behavior when PointGuardAI is unreachable. 'fail_closed' raises an error "
            "(default); 'fail_open' logs a critical error and allows the request."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "PointGuard AI"
