from typing import Optional

from pydantic import Field

from litellm.types.guardrails import GuardrailParamUITypes

from .base import GuardrailConfigModel


class AktoConfigModel(GuardrailConfigModel):
    """Configuration model for Akto Guardrail integration."""

    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for Akto authentication. "
            "If not provided, falls back to AKTO_API_KEY environment variable."
        ),
    )

    api_base: Optional[str] = Field(
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

    sync_mode: Optional[bool] = Field(
        default=True,
        description=(
            "Operating mode for the guardrail.\\n"
            "• true (sync/blocking): Pre-call guardrails check + post-call data ingestion.\\n"
            "• false (async/non-blocking): Single post-call call with guardrails + data ingestion (log-only).\\n"
            "Falls back to AKTO_SYNC_MODE environment variable."
        ),
        json_schema_extra={"ui_type": GuardrailParamUITypes.BOOL},
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

    unreachable_fallback: Optional[str] = Field(
        default="fail_closed",
        description=(
            "Behavior when the Akto service is unreachable. "
            "'fail_open' allows requests through; 'fail_closed' blocks them."
        ),
        json_schema_extra={
            "examples": ["fail_open", "fail_closed"],
        },
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Akto"
