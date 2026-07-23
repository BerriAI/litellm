"""Live e2e: POST /v1/messages (Anthropic Messages API) returns a real completion.

Registers an Anthropic deployment at runtime, drives the Messages endpoint through
the gateway, and asserts an assistant message with text came back, both
non-streaming and streamed. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import require_successful_call, unwrap
from endpoints_client import EndpointsClient, MessagesResult
from lifecycle import ResourceManager
from models import (
    AnthropicCustomTool,
    AnthropicMessagesBody,
    ChatMessage,
    JsonSchemaProperty,
    LiteLLMParamsBody,
    SpendLogRow,
    ToolInputSchema,
)

pytestmark = pytest.mark.e2e

ANTHROPIC_BACKEND = "anthropic/claude-haiku-4-5"

WEATHER_TOOL = AnthropicCustomTool(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema=ToolInputSchema(
        properties={"city": JsonSchemaProperty(type="string")},
        required=["city"],
    ),
)


def _approx_equal(actual: float, expected: float) -> bool:
    """Within 1% or 1e-9 absolute - spend math, not exact float identity."""
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


class TestAnthropicMessages:
    def _register(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> tuple[str, str]:
        model = f"e2e-messages-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"
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

    @pytest.mark.covers("llm.messages.anthropic.basic.nonstream.cost_logged")
    def test_messages_logs_cost_matching_the_response_header(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        require_env("ANTHROPIC_API_KEY")
        model = f"e2e-messages-cost-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.messages(key, model, f"reply with one word {unique_marker()}")
        require_successful_call(result)
        parsed = MessagesResult.model_validate_json(result.body)
        assert parsed.role == "assistant" and parsed.text.strip(), (
            f"/v1/messages returned no assistant text: {result.body[:300]}"
        )

        # The customer reads per-request cost off the response header (LIT-4076), so
        # it must be present and positive on /v1/messages, not only /chat/completions.
        header_cost = result.response_cost
        assert header_cost is not None and header_cost > 0, (
            "x-litellm-response-cost header missing or non-positive on /v1/messages; "
            f"headers={result.headers}"
        )

        # Correlate the spend row by the unique scoped key, not the Anthropic response
        # id: on /v1/messages the spend-log request_id is the proxy's own call id, which
        # need not equal the message body id, so an id-based poll can miss a correctly
        # logged row and time out. The key is fresh per test, so its only priced row is
        # this call.
        def _priced(rows: list[SpendLogRow]) -> bool:
            return any(r.spend is not None and r.spend > 0 for r in rows)

        rows = endpoints_client.proxy.poll_logs_for_key(key, predicate=_priced)
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
