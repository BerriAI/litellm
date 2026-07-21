from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class DeepKeepGuardrailConfigModelOptionalParams(BaseModel):
    unreachable_fallback: Optional[str] = Field(
        default="fail_closed",
        description=(
            "Behavior when the DeepKeep API is unreachable. "
            "'fail_closed' raises an error (default). 'fail_open' logs a critical "
            "error and allows the request to proceed."
        ),
    )


class DeepKeepGuardrailConfigModel(GuardrailConfigModel[DeepKeepGuardrailConfigModelOptionalParams]):
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "The API key for the DeepKeep AI Firewall. "
            "If not provided, the `DEEPKEEP_API_KEY` environment variable is checked."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "The API base URL for the DeepKeep AI Firewall. "
            "If not provided, the `DEEPKEEP_API_BASE` environment variable is checked."
        ),
    )
    deepkeep_firewall_id: Optional[str] = Field(
        default=None,
        description=(
            "The DeepKeep Firewall ID to use for guardrail evaluation. "
            "If not provided, the `DEEPKEEP_FIREWALL_ID` environment variable is checked."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "DeepKeep AI Firewall"
