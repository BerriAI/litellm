"""Live e2e: Cohere via OpenAI-compatible /chat/completions."""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

BACKEND = "cohere/command-r-08-2024"


class TestCohereChat:
    @pytest.mark.covers(
        "llm.chat_completions.cohere.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_cohere_chat_returns_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        # Pass the host env key into the deployment: the proxy container may not
        # have COHERE_API_KEY as an os.environ ref even when the test process does.
        (cohere_key,) = require_env("COHERE_API_KEY")
        model = f"e2e-cohere-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=BACKEND, api_key=cohere_key),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. {unique_marker()}",
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"cohere chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content and content.strip(), f"cohere empty content: {response}"
