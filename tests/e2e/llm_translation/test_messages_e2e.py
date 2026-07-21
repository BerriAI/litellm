"""Live e2e: POST /v1/messages (Anthropic Messages API) returns a real completion.

Registers an Anthropic deployment at runtime, drives the Messages endpoint through
the gateway, and asserts an assistant message with text came back, both
non-streaming and streamed. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call, unwrap
from endpoints_client import EndpointsClient, MessagesResult
from lifecycle import ResourceManager
from models import (
    AnthropicCustomTool,
    AnthropicMessagesBody,
    ChatMessage,
    JsonSchemaProperty,
    LiteLLMParamsBody,
    ToolInputSchema,
)

pytestmark = pytest.mark.e2e

WEATHER_TOOL = AnthropicCustomTool(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema=ToolInputSchema(
        properties={"city": JsonSchemaProperty(type="string")},
        required=["city"],
    ),
)


class TestAnthropicMessages:
    def _register(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> tuple[str, str]:
        model = f"e2e-messages-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="anthropic/claude-haiku-4-5", api_key="os.environ/ANTHROPIC_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        return model, resources.key()

    @pytest.mark.covers("llm.messages.anthropic.basic.nonstream.works")
    def test_messages_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = self._register(endpoints_client, resources)

        result = endpoints_client.messages(key, model, "reply with one word")
        require_successful_call(result)
        parsed = MessagesResult.model_validate_json(result.body)
        assert parsed.role == "assistant", f"unexpected role: {result.body[:300]}"
        assert parsed.text.strip(), f"/v1/messages returned no text: {result.body[:300]}"

    @pytest.mark.covers("llm.messages.anthropic.basic.stream.works")
    def test_messages_streams_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = self._register(endpoints_client, resources)

        result = endpoints_client.proxy.messages_stream(
            key,
            AnthropicMessagesBody(
                model=model,
                max_tokens=64,
                stream=True,
                messages=[ChatMessage(role="user", content="Count from one to three.")],
            ),
        )
        require_successful_call(result)
        assert result.is_streaming, f"response was not streamed: {result.headers}"
        assert not result.stream_error, f"stream errored: {result.stream_error}"
        assert result.stream_events, "stream produced no SSE events"
        assert any("content_block_delta" in event for event in result.stream_events), (
            "stream carried no content deltas"
        )
        assert any("message_stop" in event for event in result.stream_events), (
            "stream never reached message_stop"
        )

    @pytest.mark.covers("llm.messages.anthropic.tool_use.nonstream.works")
    def test_messages_tool_use(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = self._register(endpoints_client, resources)

        response = unwrap(
            endpoints_client.proxy.messages(
                key,
                AnthropicMessagesBody(
                    model=model,
                    max_tokens=256,
                    tools=[WEATHER_TOOL],
                    messages=[
                        ChatMessage(role="user", content="What is the weather in Paris? Use the tool.")
                    ],
                ),
            )
        )
        assert response.content, f"no content blocks in response: {response}"
        assert any(block.type == "tool_use" for block in response.content), (
            f"model did not call the tool: {response}"
        )
