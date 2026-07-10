"""Live regression net for /chat/completions across the configured providers.

GH #28991 broke /chat/completions (and /responses) for most models on some
releases: a clean 200 came back but with no real completion. A status check
alone would not have caught it, so each case here asserts the product promise -
a non-empty assistant message and a real model name in the body - across the
three providers wired into the gateway config (OpenAI, Anthropic, Gemini). A
regression that empties the completion for any provider fails that provider's
row here.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from models import ChatBody, ChatMessage
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

CHAT_MODELS: tuple[tuple[str, str], ...] = (
    ("gpt-5.5", "openai"),
    ("claude-haiku-4-5", "anthropic"),
    ("gemini-2.5-flash", "gemini"),
)


class TestChatCompletionsRegression:
    @pytest.mark.parametrize(
        ("model", "route"),
        CHAT_MODELS,
        ids=[f"{model}-{route}" for model, route in CHAT_MODELS],
    )
    @pytest.mark.covers(
        "llm.chat_completions.openai.basic.nonstream.works",
        "llm.chat_completions.anthropic.basic.nonstream.works",
        "llm.chat_completions.vertex.basic.nonstream.works",
        exercised_on=[],
    )
    def test_chat_returns_real_completion(
        self, client: PassthroughClient, scoped_key: str, model: str, route: str
    ) -> None:
        response = unwrap(
            client.gateway.chat(
                scoped_key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"reply with one word {unique_marker()}",
                        )
                    ],
                    max_tokens=512,
                ),
            )
        )

        assert (
            response.model
        ), f"{model} ({route}): response carried no model name: {response}"
        assert (
            response.choices
        ), f"{model} ({route}): response had no choices: {response}"
        message = response.choices[0].message
        assert (
            message is not None and message.content and message.content.strip()
        ), f"{model} ({route}): 200 with an empty completion (#28991): {response}"
