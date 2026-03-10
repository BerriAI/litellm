from typing import Optional, Literal

from pydantic import Field

from litellm.types.guardrails import GuardrailParamUITypes

from .base import GuardrailConfigModel


class AktoConfigModel(GuardrailConfigModel):
    """Configuration model for Akto Guardrail integration."""

    akto_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Akto Data Ingestion Service URL. "
            "Falls back to AKTO_DATA_INGESTION_URL environment variable."
        ),
        json_schema_extra={
            "examples": [
                "http://localhost:9090",
                "https://akto-ingestion.example.com",
            ]
        },
    )

    akto_api_key: Optional[str] = Field(
        default=None,
        description=(
            "Akto API key for authentication. "
            "Falls back to AKTO_API_KEY environment variable."
        ),
    )

    on_flagged: Optional[Literal["block", "monitor"]] = Field(
        default="block",
        description=(
            "Action to take when a violation occurs:\n"
            "• 'block' (sync): Pre-call validation that blocks requests if flagged.\n"
            "• 'monitor' (async): Post-call non-blocking logging only.\n"
            "Falls back to AKTO_ON_FLAGGED environment variable."
        ),
    )

    akto_account_id: Optional[str] = Field(
        default=None,
        description=(
            "Akto account ID for data ingestion. "
            "Falls back to AKTO_ACCOUNT_ID environment variable. Defaults to '1000000'."
        ),
    )

    akto_vxlan_id: Optional[str] = Field(
        default=None,
        description=(
            "Akto VXLAN ID for traffic identification. "
            "Falls back to AKTO_VXLAN_ID environment variable. Defaults to '0'."
        ),
    )

    unreachable_fallback: Optional[Literal["fail_closed", "fail_open"]] = Field(
        default="fail_closed",
        description=(
            "Behavior when the Akto service is unreachable. "
            "'fail_open' allows requests through; 'fail_closed' blocks them."
        ),
    )

    guardrail_timeout: Optional[int] = Field(
        default=None,
        description=(
            "HTTP timeout in seconds for calls to the Akto service. "
            "Defaults to 5 seconds if not set."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Akto"
