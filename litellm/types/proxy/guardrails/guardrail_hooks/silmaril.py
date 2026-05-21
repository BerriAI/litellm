from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from .base import GuardrailConfigModel


class SilmarilGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "Silmaril Firewall API key. Use os.environ/SILMARIL_API_KEY in "
            "config.yaml to load this from the LiteLLM proxy environment."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Silmaril Firewall guardrail URL. Set SILMARIL_GUARDRAIL_URL or "
            "api_base to the full /beta/litellm_basic_guardrail_api endpoint."
        ),
    )
    additional_provider_specific_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional Silmaril-specific parameters sent with each guardrail request.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_open",
        description=(
            "Behavior when Silmaril is unreachable. Defaults to 'fail_open' "
            "to allow the request to proceed; 'fail_closed' raises an error."
        ),
    )
    extra_headers: Optional[List[str]] = Field(
        default=None,
        description="Client request headers whose values should be forwarded to Silmaril.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Silmaril Firewall"
