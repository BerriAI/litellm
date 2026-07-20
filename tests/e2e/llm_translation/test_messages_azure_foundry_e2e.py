"""Live e2e: POST /v1/messages routed to Azure AI Foundry Anthropic deployments.

Registers `azure_ai/<claude>` deployments at runtime and drives the Messages
endpoint through the gateway across the behaviors an Anthropic client relies on:
a basic completion, a streamed completion, and tool use (non-streaming and
streaming). Auth is the Azure API key (`x-api-key`); the deployment reads
`AZURE_AI_API_BASE` / `AZURE_AI_API_KEY` from the proxy env, so no secret is
sent in the request.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call, unwrap
from endpoints_client import EndpointsClient
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

AZURE_FOUNDRY_MODEL = "azure_ai/claude-haiku-4-5"

WEATHER_TOOL = AnthropicCustomTool(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema=ToolInputSchema(
        properties={"city": JsonSchemaProperty(type="string")},
        required=["city"],
    ),
)


def _assert_streamed_ok(result: StreamingResponse) -> None:
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


class TestAzureFoundryMessages:
    def _register(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> tuple[str, str]:
        model = f"e2e-azure-foundry-messages-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model=AZURE_FOUNDRY_MODEL,
                api_base="os.environ/AZURE_AI_API_BASE",
                api_key="os.environ/AZURE_AI_API_KEY",
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        return model, resources.key(models=[model])

    @pytest.mark.covers("llm.messages.azure_foundry.basic.nonstream.works")
    def test_basic_nonstream(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = self._register(endpoints_client, resources)
        response = unwrap(
            endpoints_client.proxy.messages(
                key,
                AnthropicMessagesBody(
                    model=model,
                    max_tokens=64,
                    messages=[ChatMessage(role="user", content="Reply with one word.")],
                ),
            )
        )
        assert response.content, f"no content blocks in response: {response}"
        text = "".join(block.text or "" for block in response.content if block.type == "text")
        assert text.strip(), f"/v1/messages returned no text: {response}"

    @pytest.mark.covers("llm.messages.azure_foundry.basic.stream.works")
    def test_basic_stream(
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
        _assert_streamed_ok(result)

    @pytest.mark.covers("llm.messages.azure_foundry.tool_use.nonstream.works")
    def test_tool_use_nonstream(
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

    @pytest.mark.covers("llm.messages.azure_foundry.tool_use.stream.works")
    def test_tool_use_stream(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = self._register(endpoints_client, resources)
        result = endpoints_client.proxy.messages_stream(
            key,
            AnthropicMessagesBody(
                model=model,
                max_tokens=256,
                stream=True,
                tools=[WEATHER_TOOL],
                messages=[
                    ChatMessage(role="user", content="What is the weather in Paris? Use the tool.")
                ],
            ),
        )
        require_successful_call(result)
        assert result.is_streaming, f"response was not streamed: {result.headers}"
        assert not result.stream_error, f"stream errored: {result.stream_error}"
        assert result.stream_events, "stream produced no SSE events"
        assert any("tool_use" in event for event in result.stream_events), (
            "stream carried no tool_use block"
        )
        assert any("message_stop" in event for event in result.stream_events), (
            "stream never reached message_stop"
        )
