"""Live e2e for output-token limits on Bedrock Mantle GPT-5.6.

Mantle serves the OpenAI GPT-5.x models on its OpenAI-compatible endpoint,
which rejects max_tokens for them and only accepts max_completion_tokens. A
client that sets either output limit must get a completion back, not the
upstream 400 "Unsupported parameter: 'max_tokens' is not supported with this
model."
"""

from __future__ import annotations

from typing import Final

import pytest
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

MANTLE_BACKEND: Final = "bedrock_mantle/openai.gpt-5.6-terra"
OUTPUT_LIMIT: Final = 256


def _register_mantle_model(client: PassthroughClient, resources: ResourceManager) -> str:
    model = f"e2e-mantle-output-limit-{unique_marker()}"
    model_id = client.proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=MANTLE_BACKEND,
            api_key="os.environ/AWS_BEARER_TOKEN_BEDROCK",
            aws_region_name="us-east-1",
        ),
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model


def _prompt() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="reply with one word")]


class TestBedrockMantleOutputLimit:
    def test_max_completion_tokens_is_accepted(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model = _register_mantle_model(client, resources)
        key = resources.key()

        result = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(key),
            json=ChatBody(model=model, messages=_prompt(), max_completion_tokens=OUTPUT_LIMIT),
        )

        assert result.ok, f"chat call failed: {result.status_code} {result.body[:400]}"

    def test_max_tokens_is_accepted(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model = _register_mantle_model(client, resources)
        key = resources.key()

        result = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(key),
            json=ChatBody(model=model, messages=_prompt(), max_tokens=OUTPUT_LIMIT),
        )

        assert result.ok, f"chat call failed: {result.status_code} {result.body[:400]}"
