import enum

from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class HiddenlayerAction(str, enum.Enum):
    BLOCK = "Block"
    REDACT = "Redact"


class HiddenlayerMessages(str, enum.Enum):
    BLOCK_MESSAGE = "Blocked by Hiddenlayer."


class HiddenlayerGuardrailConfigModel(GuardrailConfigModel):
    api_base: Optional[str] = Field(
        default=None,
        description="The URL of the Hiddenlayer server. If not provided, the `HIDDENLAYER_API_BASE` environment variable is checked or https://api.hiddenlayer.ai is used.",
    )

    api_id: Optional[str] = Field(
        default=None,
        description="The Hiddenlayer API Id for the Hiddenlayer API. If not provided, the `HIDDENLAYER_CLIENT_ID` environment variable is checked or https://api.hiddenlayer.ai is used.",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="The Hiddenlayer Secret Key for the Hiddenlayer API.. If not provided, the `HIDDENLAYER_CLIENT_SECRET` environment variable is checked.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Hiddenlayer Guardrail"
