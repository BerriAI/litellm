from typing import Optional, Literal

from pydantic import Field

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

class ThirdlawGuardrailConfigModel(GuardrailConfigModel):
    api_base: Optional[str] = Field(
        default=None,
        description="ThirdLaw Guardrail API Base URL. Env: THIRDLAW_API_BASE.",
        json_schema_extra={
            "examples": [
                "https://api.thirdlaw.<your-domain>",
            ]
        },
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for ThirdLaw. Env: THIRDLAW_API_KEY.",
    )

    additional_headers: Optional[str] = Field(
        default=None,
        description="Comma-separated list of inbound request header names whose values ThirdLaw should receive. All inbound headers are forwarded, but only headers listed here have their actual values exposed; others are forwarded as [present]. Example: x-request-id,x-correlation-id.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description="Controls LiteLLM behavior when ThirdLaw is unavailable or returns a non-policy error. fail_closed blocks the request with 500. fail_open allows the request to continue. A block decision from ThirdLaw always returns 400 regardless of this setting.",
    )

    guardrail_timeout: Optional[int] = Field(
        default=60,
        description="Timeout for the ThirdLaw API request. In seconds.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "ThirdLaw"
