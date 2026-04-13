from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class PromptGuardConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for PromptGuard authentication. "
            "If not provided, the PROMPTGUARD_API_KEY "
            "environment variable is used."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "PromptGuard API base URL. "
            "Defaults to https://api.promptguard.co. "
            "Falls back to PROMPTGUARD_API_BASE env var."
        ),
    )
    block_on_error: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to block the request when the "
            "PromptGuard API is unreachable. "
            "Defaults to true (fail-closed). "
            "Set to false for fail-open behaviour."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "PromptGuard"
