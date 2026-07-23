"""Live e2e: POST /v1/messages (Anthropic Messages API) returns a real completion.

Registers an Anthropic deployment at runtime and drives the Messages endpoint
through the gateway with the real Anthropic SDK, the client customers actually
use (LIT-4577), asserting an assistant message with text came back, both
non-streaming and streamed.
"""

from __future__ import annotations

import pytest
from anthropic.types import Message, ToolParam

from e2e_config import require_env, unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, SpendLogRow
from proxy_client import ProxyClient
from sdk_clients import SdkClients, response_header

pytestmark = pytest.mark.e2e

ANTHROPIC_BACKEND = "anthropic/claude-haiku-4-5"

WEATHER_TOOL: ToolParam = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _approx_equal(actual: float, expected: float) -> bool:
    """Within 1% or 1e-9 absolute - spend math, not exact float identity."""
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


def _text(message: Message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


class TestAnthropicMessages:
    def _register(
        self, proxy: ProxyClient, resources: ResourceManager, prefix: str = "e2e-messages"
    ) -> str:
        model = f"{prefix}-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        return model

    @pytest.mark.covers("llm.messages.anthropic.basic.nonstream.works")
    def test_messages_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = self._register(proxy, resources)
        client = sdk.anthropic(resources.key())

        message = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "reply with one word"}],
        )
        assert message.role == "assistant", f"unexpected role: {message.role!r}"
        assert _text(message).strip(), f"/v1/messages returned no text: {message.content!r}"

    @pytest.mark.covers("llm.messages.anthropic.basic.nonstream.cost_logged")
    def test_messages_logs_cost_matching_the_response_header(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        require_env("ANTHROPIC_API_KEY")
        model = self._register(proxy, resources, prefix="e2e-messages-cost")
        key = resources.key()
        client = sdk.anthropic(key)

        raw = client.messages.with_raw_response.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": f"reply with one word {unique_marker()}"}],
        )
        message = raw.parse()
        assert message.role == "assistant" and _text(message).strip(), (
            f"/v1/messages returned no assistant text: {message.content!r}"
        )

        # The customer reads per-request cost off the response header (LIT-4076), so
        # it must be present and positive on /v1/messages, not only /chat/completions.
        raw_header_cost = response_header(raw.headers, "x-litellm-response-cost")
        assert raw_header_cost is not None, (
            "x-litellm-response-cost header missing on /v1/messages; "
            f"headers={dict(raw.headers)}"
        )
        header_cost = float(raw_header_cost)
        assert header_cost > 0, (
            f"x-litellm-response-cost header non-positive on /v1/messages: {header_cost}"
        )

        # Correlate the spend row by the unique scoped key, not the Anthropic response
        # id: on /v1/messages the spend-log request_id is the proxy's own call id, which
        # need not equal the message body id, so an id-based poll can miss a correctly
        # logged row and time out. The key is fresh per test, so its only priced row is
        # this call.
        def _priced(rows: list[SpendLogRow]) -> bool:
            return any(r.spend is not None and r.spend > 0 for r in rows)

        rows = proxy.poll_logs_for_key(key, predicate=_priced)
        priced = [r for r in rows if r.spend is not None and r.spend > 0]
        assert priced, (
            f"no priced /spend/logs row landed for key {key} within the poll window; got {rows}"
        )
        row = priced[0]
        assert (row.prompt_tokens or 0) > 0 and (row.completion_tokens or 0) > 0, (
            f"messages spend row missing token counts, so the cost is not real usage: {row}"
        )
        assert row.spend is not None and _approx_equal(row.spend, header_cost), (
            f"logged spend {row.spend} disagrees with the x-litellm-response-cost header {header_cost}; "
            "the customer bills against the header, so the two must match"
        )

    @pytest.mark.covers("llm.messages.anthropic.basic.stream.works")
    def test_messages_streams_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = self._register(proxy, resources)
        client = sdk.anthropic(resources.key())

        stream = client.messages.create(
            model=model,
            max_tokens=64,
            stream=True,
            messages=[{"role": "user", "content": "Count from one to three."}],
        )
        event_types = [event.type for event in stream]
        assert event_types, "stream produced no SSE events"
        assert "content_block_delta" in event_types, "stream carried no content deltas"
        assert "message_stop" in event_types, "stream never reached message_stop"

    @pytest.mark.covers("llm.messages.anthropic.tool_use.nonstream.works")
    def test_messages_tool_use(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = self._register(proxy, resources)
        client = sdk.anthropic(resources.key())

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
