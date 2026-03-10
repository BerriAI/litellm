from typing import Optional, Literal

from pydantic import Field

from litellm.types.guardrails import GuardrailParamUITypes

from .base import GuardrailConfigModel


class AktoConfigModel(GuardrailConfigModel):
    """Config for the Akto guardrail."""

    akto_base_url: Optional[str] = Field(
        default=None,
        description="Akto Guardrail API Base URL. Env: AKTO_GUARDRAIL_API_BASE.",
        json_schema_extra={
            "examples": [
                "http://localhost:9090",
                "https://akto-ingestion.example.com",
            ]
        },
    )

    akto_api_key: Optional[str] = Field(
        default=None,
        description="API key for Akto. Env: AKTO_API_KEY.",
    )

    on_flagged: Optional[Literal["block", "monitor"]] = Field(
        default="block",
        description="'block' = pre-call validation, 'monitor' = post-call log only. Env: AKTO_ON_FLAGGED.",
    )

    unreachable_fallback: Optional[Literal["fail_closed", "fail_open"]] = Field(
        default="fail_closed",
        description="What to do when Akto is unreachable. 'fail_open' = allow, 'fail_closed' = block.",
    )

    guardrail_timeout: Optional[int] = Field(
        default=None,
        description="HTTP timeout in seconds. Default: 5.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Akto"
