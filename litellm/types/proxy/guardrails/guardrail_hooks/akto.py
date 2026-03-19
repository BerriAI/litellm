from typing import Optional, Literal

from pydantic import Field

from .base import GuardrailConfigModel


class AktoConfigModel(GuardrailConfigModel):
    """Config for the Akto guardrail.

    Modes:
      pre_call       -> validate request, block if flagged
      logging_only   -> non-blocking ingestion of request+response
      [pre_call, logging_only] -> both (recommended)
    """

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

    akto_account_id: Optional[str] = Field(
        default=None,
        description="Akto account ID for multi-tenant deployments. Env: AKTO_ACCOUNT_ID. Default: '1000000'.",
    )

    akto_vxlan_id: Optional[str] = Field(
        default=None,
        description="Akto VXLAN ID. Env: AKTO_VXLAN_ID. Default: '0'.",
    )

    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description="What to do when Akto is unreachable. 'fail_open' = allow, 'fail_closed' = block.",
    )

    guardrail_timeout: Optional[int] = Field(
        default=None,
        description="HTTP timeout in seconds for pre_call validation. Default: 5.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Akto"
