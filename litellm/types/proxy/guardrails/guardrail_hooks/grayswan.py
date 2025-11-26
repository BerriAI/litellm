"""Gray Swan guardrail configuration models."""

from typing import Dict, Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class GraySwanGuardrailConfigModelOptionalParams(BaseModel):
    """Optional parameters for the Gray Swan guardrail."""

    on_flagged_action: Optional[str] = Field(
        default="monitor",
        description="Action when a violation is detected: 'block' rejects the call, 'monitor' logs only, 'passthrough' includes detection info in response without blocking.",
    )
    violation_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold between 0 and 1 at which Gray Swan violations trigger the configured action.",
    )
    reasoning_mode: Optional[str] = Field(
        default=None,
        description="Gray Swan reasoning mode override. Accepted values: 'off', 'hybrid', 'thinking'.",
    )
    policy_id: Optional[str] = Field(
        default=None,
        description="Gray Swan policy identifier to apply during monitoring.",
    )
    categories: Optional[Dict[str, str]] = Field(
        default=None,
        description="Default Gray Swan category definitions to send with each request.",
    )


class GraySwanGuardrailConfigModel(
    GuardrailConfigModel[GraySwanGuardrailConfigModelOptionalParams]
):
    """Configuration parameters for the Gray Swan guardrail."""

    api_key: Optional[str] = Field(
        default=None,
        description="API key for Gray Swan. Reads from the `GRAYSWAN_API_KEY` environment variable when omitted.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Override for the Gray Swan API base URL. Defaults to https://api.grayswan.ai and can be set via `GRAYSWAN_API_BASE`.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Gray Swan Guardrail"
