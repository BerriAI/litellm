from pydantic import Field

from .base import GuardrailConfigModel


class AliceGuardrailConfigModel(GuardrailConfigModel):
    api_key: str | None = Field(
        default=None,
        description=(
            "The API key for Alice by ActiveFence. "
            "If not provided, the `ALICE_API_KEY` environment variable is checked."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "The API base URL for Alice. If not provided, the `ALICE_API_BASE` environment "
            "variable is checked, then `https://api.alice.io`."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Alice by ActiveFence"
