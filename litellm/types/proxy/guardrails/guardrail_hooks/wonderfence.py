"""Alice WonderFence guardrail configuration models."""

from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class WonderFenceGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Alice WonderFence guardrail."""

    api_key: Optional[str] = Field(
        default=None,
        description="API key for WonderFence. Can also be set via WONDERFENCE_API_KEY.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Override for WonderFence API base URL.",
    )
    app_name: Optional[str] = Field(
        default="litellm",
        description="Application name for WonderFence. Can also be set via WONDERFENCE_APP_NAME.",
    )
    api_timeout: Optional[float] = Field(
        default=10.0,
        description="Timeout in seconds for API calls.",
    )
    platform: Optional[str] = Field(
        default=None,
        description="Cloud platform (e.g., aws, azure, databricks).",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Alice WonderFence Guardrail"
