from __future__ import annotations

from typing import Final

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel

from .base import GuardrailConfigModel

DEFAULT_API_BASE: Final = "https://app.collieai.io/v1/integrations/litellm"


class CollieAiGuardrailConfigModelOptionalParams(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    deadline_ms: int | None = Field(
        default=2500,
        gt=0,
        description="Each phase's whole deadline in milliseconds: connection, send, reply and parse.",
    )
    forward_conversation_id: bool | None = Field(
        default=False,
        description="Forward the caller's x-collieai-conversation-id header, and only that header.",
    )


class CollieAiGuardrailConfigModel(GuardrailConfigModel[CollieAiGuardrailConfigModelOptionalParams]):
    api_key: str = Field(
        min_length=1,
        description="CollieAi project API key, starting with clai_. Env: COLLIEAI_API_KEY.",
    )
    api_base: str | None = Field(
        default=DEFAULT_API_BASE,
        description="CollieAi's LiteLLM integration base URL.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "CollieAi"
