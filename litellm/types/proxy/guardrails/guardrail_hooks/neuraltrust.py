from typing import Final, Literal

from pydantic import Field

from .base import GuardrailConfigModel

DEFAULT_API_BASE: Final = "https://trustguard.neuraltrust.ai"


class NeuralTrustGuardrailConfigModel(GuardrailConfigModel):
    """Config for the NeuralTrust TrustGuard native LiteLLM hook."""

    api_key: str | None = Field(
        default=None,
        description=("TrustGuard API key (tgk_...). If not provided, TRUSTGUARD_API_KEY is checked."),
    )

    api_base: str | None = Field(
        default=None,
        description=("TrustGuard API base URL. Default https://trustguard.neuraltrust.ai. Env: TRUSTGUARD_API_BASE."),
    )

    collector_key: str | None = Field(
        default=None,
        description=(
            "TrustGuard collector key (tgcol_...). Optional when the API key is bound to a "
            "collector. Env: TRUSTGUARD_COLLECTOR_KEY."
        ),
    )

    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description=(
            "What to do on transport failures (connect errors, timeouts, HTTP 502/504). "
            "'fail_closed' blocks the request; 'fail_open' allows it. "
            "HTTP 503 entitlements, 401/403, other 4xx/5xx, unknown verdicts, and "
            "unusable transform payloads always fail closed. "
            "'fail_open' means the request bypasses TrustGuard entirely."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "NeuralTrust"
