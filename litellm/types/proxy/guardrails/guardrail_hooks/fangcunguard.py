from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class FangcunGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The API key for FangcunGuard. If not provided, the `FANGCUN_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The API base for FangcunGuard. Defaults to https://api.fangcunleap.com; the `FANGCUN_API_BASE` environment variable is checked if set.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "FangcunGuard"
