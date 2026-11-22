from __future__ import annotations

from typing import Final

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel

from .base import GuardrailConfigModel

DEFAULT_API_BASE: Final = "https://app.collieai.io/v1/integrations/litellm"


class CollieAiGuardrailConfigModelOptionalParams(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    deadline_ms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Each phase's whole deadline in milliseconds: connection, send, reply and parse. "
            "Leave empty for the package default."
        ),
    )
    forward_conversation_id: bool | None = Field(
        default=None,
        description=(
            "Forward the caller's x-collieai-conversation-id header, and only that header. Leave empty to keep it off."
        ),
    )


class CollieAiGuardrailConfigModel(GuardrailConfigModel[CollieAiGuardrailConfigModelOptionalParams]):
    model_config = ConfigDict(frozen=True)

    api_key: str | None = Field(
        default=None,
        description="CollieAi project API key, starting with clai_. Leave empty to use COLLIEAI_API_KEY.",
    )
    api_base: str | None = Field(
        default=None,
        description="CollieAi's LiteLLM integration base URL. Leave empty for CollieAi's hosted service.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "CollieAi"
