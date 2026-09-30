"""Live e2e for output-token limits on a Bedrock Mantle GPT-5.6 chat deployment.

A deployment registered with model_info.mode "chat" and an explicit Mantle
OpenAI-compatible api_base is served through Mantle chat completions. Mantle
rejects max_tokens for the GPT-5.x models there and only accepts
max_completion_tokens, so a caller that sets an output limit, or a deployment
that stores one, must still get a completion back instead of the upstream 400
"Unsupported parameter: 'max_tokens' is not supported with this model."
"""

from __future__ import annotations

from typing import Final

import pytest
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    ChatStreamOptions,
    ChatTool,
    ChatToolFunction,
    LiteLLMParamsBody,
    ModelInfoBody,
    ModelNewBody,
)
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

MANTLE_BACKEND: Final = "bedrock_mantle/openai.gpt-5.6-terra"
MANTLE_CHAT_API_BASE: Final = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
OUTPUT_LIMIT: Final = 1000
WEATHER_TOOL: Final = ChatTool(
    function=ChatToolFunction(
        name="get_weather",
        description="Get the weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
)


def _register_mantle_chat_model(
    client: PassthroughClient, resources: ResourceManager, max_tokens: int | None = None
) -> str:
    model = f"e2e-mantle-chat-output-limit-{unique_marker()}"
    model_id = client.proxy.register_model(
        ModelNewBody(
            model_name=model,
            litellm_params=LiteLLMParamsBody(
                model=MANTLE_BACKEND,
                custom_llm_provider="bedrock_mantle",
                api_key="os.environ/AWS_BEARER_TOKEN_BEDROCK",
                api_base=MANTLE_CHAT_API_BASE,
                reasoning_effort="none",
                max_tokens=max_tokens,
            ),
            model_info=ModelInfoBody(mode="chat", supported_endpoints=["/v1/chat/completions", "/v1/responses"]),
        )
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model


def _prompt() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="reply with one word")]


class TestBedrockMantleChatOutputLimit:
    def test_client_max_tokens_returns_completion(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model = _register_mantle_chat_model(client, resources)
        result = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(resources.key()),
            json=ChatBody(model=model, messages=_prompt(), max_tokens=OUTPUT_LIMIT),
        )
        assert result.ok, f"chat call failed: {result.status_code} {result.body[:400]}"

    def test_client_max_completion_tokens_returns_completion(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_mantle_chat_model(client, resources)
        result = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(resources.key()),
            json=ChatBody(model=model, messages=_prompt(), max_completion_tokens=OUTPUT_LIMIT),
        )
        assert result.ok, f"chat call failed: {result.status_code} {result.body[:400]}"

    def test_deployment_max_tokens_streams_tool_call_completion(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_mantle_chat_model(client, resources, max_tokens=OUTPUT_LIMIT)
        result = client.proxy.chat_stream(
            resources.key(),
            ChatBody(
                model=model,
                messages=_prompt(),
                stream=True,
                stream_options=ChatStreamOptions(include_usage=True),
                tools=[WEATHER_TOOL],
            ),
        )
        assert result.ok and result.stream_error is None and result.stream_done, (
            f"streamed chat call failed: {result.status_code} {result.stream_error} {result.body[:400]}"
        )
