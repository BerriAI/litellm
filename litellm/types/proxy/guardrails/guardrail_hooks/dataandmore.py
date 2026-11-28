from typing import Literal

from pydantic import Field

from .base import GuardrailConfigModel


class DataAndMoreGuardrailConfigModel(GuardrailConfigModel):  # pyright: ignore[reportMissingTypeArgument]  # parameterizing adds an optional_params field to the UI form
    """Configuration parameters for the Data & More DLP guardrail."""

    api_key: str | None = Field(
        default=None,
        description=(
            "Data & More AI gateway guardrail key (dgmg_*). "
            "If not provided, the DGM_GUARDRAIL_KEY environment variable is checked."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Required. Your Data & More (DGM) host plus /api/dlp/ai-gateway/litellm, "
            "e.g. https://dgm.dataandmore.com/api/dlp/ai-gateway/litellm. "
            "LiteLLM appends /beta/litellm_basic_guardrail_api. "
            "Override with DGM_GUARDRAIL_API_BASE when unset in config."
        ),
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_open",
        description=(
            "Behavior when Data & More is unreachable. "
            "fail_open lets chats continue (Data & More default). "
            "fail_closed blocks traffic when the guardrail is unavailable."
        ),
    )
    fail_on_error: bool | None = Field(
        default=True,
        description=("If False, allow requests when the guardrail returns an error other than an explicit block."),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Data & More"
