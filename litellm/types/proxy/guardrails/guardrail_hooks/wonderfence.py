"""Alice WonderFence guardrail configuration models."""

from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class WonderFenceGuardrailConfigModelOptionalParams(BaseModel):
    """Optional parameters for the Alice WonderFence guardrail."""

    api_base: Optional[str] = Field(
        default=None,
        description="Override for the WonderFence API base URL. Optional - uses SDK default if not provided.",
    )
    app_name: Optional[str] = Field(
        default="litellm",
        description="Application name for WonderFence. Defaults to 'litellm' if not provided. Can also be set via WONDERFENCE_APP_NAME environment variable.",
    )
    api_timeout: Optional[float] = Field(
        default=10.0,
        description="Timeout in seconds for WonderFence API calls. Defaults to 10.0 seconds.",
    )
    platform: Optional[str] = Field(
        default=None,
        description="Cloud platform where the model is hosted (e.g., aws, azure, databricks). Optional contextual info for WonderFence.",
    )
    retry_max: Optional[int] = Field(
        default=None,
        description="Maximum number of retries for failed API requests. Uses SDK default (3) if not provided.",
    )
    retry_base_delay: Optional[float] = Field(
        default=None,
        description="Base delay in seconds for retry backoff. Uses SDK default (1.0) if not provided.",
    )


class WonderFenceGuardrailConfigModel(
    GuardrailConfigModel[WonderFenceGuardrailConfigModelOptionalParams]
):
    """Configuration parameters for the Alice WonderFence guardrail."""

    api_key: Optional[str] = Field(
        default=None,
        description="API key for Alice WonderFence. Can also be set via WONDERFENCE_API_KEY environment variable.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Alice WonderFence Guardrail"
