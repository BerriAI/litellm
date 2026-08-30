from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from tests.route_parity.fixture_models import FixtureModel, ParityCase, SdkInputBase


class ChatMessage(FixtureModel):
    role: Literal["user"]
    content: str


class AnthropicChatCompletionSdkInput(SdkInputBase):
    model: Literal["anthropic/claude-sonnet-4-6"]
    messages: Annotated[tuple[ChatMessage, ...], Field(min_length=1)]
    max_tokens: int = Field(ge=1)
    stream: bool


class ChatCompletionParityCase(ParityCase[AnthropicChatCompletionSdkInput]):
    pass
