from typing import Literal

from pydantic import Field

from .base import GuardrailConfigModel


class WingbackGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Wingback guardrail."""

    api_key: str | None = Field(
        default=None,
        description=(
            "Wingback external gateway integration API key (wbk_eg_*). "
            "If not provided, the WINGBACK_INTEGRATION_API_KEY environment variable is checked."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Wingback connectors service base URL. LiteLLM appends "
            "/beta/litellm_basic_guardrail_api. Defaults to https://api.wingback.ai/connectors "
            "or WINGBACK_API_BASE when unset."
        ),
    )
    wingback_app_id: str | None = Field(
        default=None,
        description=(
            "Wingback external gateway integration name for request attribution "
            "(sent as additional_provider_specific_params.wingback_app_id)."
        ),
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_open",
        description=(
            "Behavior when the Wingback connectors service is unreachable. "
            "fail_open allows traffic (monitor mode); fail_closed blocks traffic (enforce mode)."
        ),
    )
    fail_on_error: bool | None = Field(
        default=True,
        description=(
            "If False, allow requests when the guardrail returns an error other than an explicit block."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Wingback"
