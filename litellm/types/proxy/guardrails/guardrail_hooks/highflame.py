from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .base import GuardrailConfigModel


class HighflameGuardrailConfigModel(GuardrailConfigModel[BaseModel]):
    model_config = ConfigDict(frozen=True)

    api_key: str | None = Field(
        default=None,
        description=(
            "The Highflame service key. If not provided, the `HIGHFLAME_API_KEY` environment variable is checked."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "The Highflame API base URL. If not provided, the `HIGHFLAME_API_BASE` environment variable is "
            "checked, then `https://api.highflame.ai`."
        ),
    )
    token_url: str | None = Field(
        default=None,
        description=(
            "The Highflame token endpoint that exchanges the service key for a short-lived access token. "
            "If not provided, the `HIGHFLAME_TOKEN_URL` environment variable is checked, then "
            "`https://auth.highflame.ai/oauth2/token`."
        ),
    )
    shield_mode: Literal["enforce", "monitor", "alert", "modify"] | None = Field(
        default=None,
        description=(
            "How Highflame Shield acts on a policy match: `enforce` (default) blocks, `monitor` and `alert` "
            "only record it, `modify` redacts sensitive data instead of blocking."
        ),
    )

    streaming_buffer_until_moderated: bool | None = Field(
        default=None,
        description=(
            "Hold every chunk of a streamed response until Highflame has checked the whole response, so "
            "blocked content never reaches the caller in part. Defaults to true; set false to stream live "
            "and check at the end."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Highflame"
