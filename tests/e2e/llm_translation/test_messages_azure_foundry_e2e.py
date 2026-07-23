"""Live e2e: POST /v1/messages routed to Azure AI Foundry Anthropic deployments.

Registers `azure_ai/<claude>` deployments at runtime and drives the Messages
endpoint through the gateway with the real Anthropic SDK (LIT-4577) across the
behaviors an Anthropic client relies on: a basic completion, a streamed
completion, and tool use (non-streaming and streaming). The deployment reads
`AZURE_AI_API_BASE` / `AZURE_AI_API_KEY` from the proxy env, so no secret is
sent in the request.
"""

from __future__ import annotations

import pytest
from anthropic.types import RawMessageStreamEvent, ToolParam

from e2e_config import EXPECT_RUST, unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

AZURE_FOUNDRY_MODEL = "azure_ai/claude-haiku-4-5"

WEATHER_TOOL: ToolParam = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _assert_rust_served(headers: dict[str, str]) -> None:
    if not EXPECT_RUST:
        return
    assert headers.get("x-litellm-rust") == "true", (
        "E2E_EXPECT_RUST is set, so this gateway must serve /v1/messages through the "
        "Rust path, but the response carried no x-litellm-rust marker. The request "
        "still succeeded, which is exactly the failure mode: a gateway whose native "
        f"extension is unavailable falls back to Python silently. headers={headers}"
    )


def _assert_streamed_ok(event_types: list[str]) -> None:
    assert event_types, "stream produced no SSE events"
    assert "content_block_delta" in event_types, "stream carried no content deltas"
    assert "message_stop" in event_types, "stream never reached message_stop"


class TestAzureFoundryMessages:
    def _register(self, proxy: ProxyClient, resources: ResourceManager) -> str:
        model = f"e2e-azure-foundry-messages-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=AZURE_FOUNDRY_MODEL,
                api_base="os.environ/AZURE_AI_API_BASE",
                api_key="os.environ/AZURE_AI_API_KEY",
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        return model

    @pytest.mark.covers("llm.messages.azure_foundry.basic.nonstream.works")
    def test_basic_nonstream(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = self._register(proxy, resources)
        client = sdk.anthropic(resources.key(models=[model]))

        message = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Reply with one word."}],
        )
        assert message.content, f"no content blocks in response: {message!r}"
        text = "".join(block.text for block in message.content if block.type == "text")
        assert text.strip(), f"/v1/messages returned no text: {message.content!r}"

    @pytest.mark.covers("llm.messages.azure_foundry.basic.stream.works")
    def test_basic_stream(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = self._register(proxy, resources)
        client = sdk.anthropic(resources.key(models=[model]))

        raw = client.messages.with_raw_response.create(
            model=model,
            max_tokens=64,
            stream=True,
            messages=[{"role": "user", "content": "Count from one to three."}],
        )
        _assert_rust_served({name.lower(): value for name, value in raw.headers.items()})
        _assert_streamed_ok([event.type for event in raw.parse()])

    @pytest.mark.covers("llm.messages.azure_foundry.tool_use.nonstream.works")
    def test_tool_use_nonstream(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = self._register(proxy, resources)
        client = sdk.anthropic(resources.key(models=[model]))

        message = client.messages.create(
            model=model,
            max_tokens=256,
            tools=[WEATHER_TOOL],
            messages=[{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        )
        assert message.content, f"no content blocks in response: {message!r}"
        assert any(block.type == "tool_use" for block in message.content), (
            f"model did not call the tool: {message.content!r}"
        )

    @pytest.mark.covers("llm.messages.azure_foundry.tool_use.stream.works")
    def test_tool_use_stream(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = self._register(proxy, resources)
        client = sdk.anthropic(resources.key(models=[model]))

        stream = client.messages.create(
            model=model,
            max_tokens=256,
            stream=True,
            tools=[WEATHER_TOOL],
            messages=[{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        )
        events: list[RawMessageStreamEvent] = list(stream)
        event_types = [event.type for event in events]
        assert event_types, "stream produced no SSE events"
        assert any(
            event.type == "content_block_start" and event.content_block.type == "tool_use"
            for event in events
        ), "stream carried no tool_use block"
        assert "message_stop" in event_types, "stream never reached message_stop"
