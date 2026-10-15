from pydantic import Field

from .base import GuardrailConfigModel


class LLMShieldProxyGuardrailConfigModel(GuardrailConfigModel):
    api_key: str | None = Field(
        default=None,
        description=(
            "The virtual key for the LLM Shield Proxy instance. If not provided, the "
            "`LLM_SHIELD_PROXY_API_KEY` environment variable is checked."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "The base URL of the LLM Shield Proxy instance. If not provided, the `LLM_SHIELD_PROXY_API_BASE` "
            "environment variable is checked, then `http://localhost:8000`."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "LLM Shield Proxy"
