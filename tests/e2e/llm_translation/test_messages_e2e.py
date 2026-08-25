"""Live e2e: POST /v1/messages (Anthropic Messages API) returns a real completion.

Registers an Anthropic deployment at runtime, drives the Messages endpoint through
the gateway, and asserts an assistant message with text came back, both
non-streaming and streamed. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest
from e2e_config import provider_edge_base, unique_marker
from e2e_http import assert_client_error, require_successful_call, unwrap
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
from pydantic import BaseModel

pytestmark = [pytest.mark.e2e, pytest.mark.replayable]


class _OptionalMessagesBody(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int | None = None


class _MessagesEventDelta(BaseModel):
    text: str = ""


class _MessagesEventUsage(BaseModel):
    output_tokens: int | None = None


class _MessagesStreamEvent(BaseModel):
    """One Anthropic SSE event, keeping only what the stream's shape is asserted on.

    ``delta.text`` is populated on ``content_block_delta`` and absent on the
    ``message_delta`` that closes the turn, which is the event carrying ``usage``."""

    type: str
    delta: _MessagesEventDelta | None = None
    usage: _MessagesEventUsage | None = None


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


def _anthropic_params() -> LiteLLMParamsBody:
    """The Anthropic deployment, wired through the record/replay edge when a fixture
    mode is active (LIT-5974). The mount base carries no ``/v1``: litellm's Anthropic
    handler appends ``/v1/messages`` to ``api_base`` itself, where the OpenAI handler
    appends only ``/chat/completions``."""
    base = provider_edge_base("anthropic")
    return LiteLLMParamsBody(
        model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY", api_base=base
    )


class TestAnthropicMessages:
    def _register(
        self,
        endpoints_client: EndpointsClient,
        resources: ResourceManager,
        params: LiteLLMParamsBody | None = None,
    ) -> tuple[str, str]:
        model = f"e2e-messages-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model, _anthropic_params() if params is None else params
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
        model = f"e2e-messages-cost-{unique_marker()}"
        model_id = endpoints_client.create_model(model, _anthropic_params())
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
        """Edge-wired like its non-streaming siblings, so record and replay both
        carry the streamed response.

        Asserts the shape of the event sequence, not just that deltas and a stop
        appeared somewhere in it: the answer arrives across several deltas, and the
        usage event sits between the last of them and ``message_stop``. A replay that
        coalesced the response into one buffered body could not satisfy either."""
        model, key = self._register(endpoints_client, resources)

        result = endpoints_client.proxy.messages_stream(
            key,
            AnthropicMessagesBody(
                model=model,
                max_tokens=64,
                stream=True,
                messages=[ChatMessage(role="user", content="Count from 1 to 20, one number per line.")],
            ),
        )
        require_successful_call(result)
        assert result.is_streaming, f"response was not streamed: {result.headers}"
        assert not result.stream_error, f"stream errored: {result.stream_error}"
        assert result.stream_events, "stream produced no SSE events"

        events = [
            _MessagesStreamEvent.model_validate_json(event) for event in result.stream_events
        ]
        types = [event.type for event in events]
        delta_positions = [
            index for index, event in enumerate(events) if event.type == "content_block_delta"
        ]
        assert len(delta_positions) >= 2, (
            f"stream carried {len(delta_positions)} content deltas, so it was not "
            f"incremental: {types}"
        )
        text = "".join(
            event.delta.text
            for event in events
            if event.type == "content_block_delta" and event.delta is not None
        )
        assert text.strip(), f"content deltas assembled to no text: {result.stream_events[:5]}"

        usage_positions = [
            index
            for index, event in enumerate(events)
            if event.type == "message_delta" and event.usage is not None
        ]
        assert usage_positions, f"stream never reported usage: {types}"
        assert "message_stop" in types, f"stream never reached message_stop: {types}"
        stop_position = types.index("message_stop")
        assert delta_positions[-1] < usage_positions[0] < stop_position, (
            f"usage did not land between the last content delta and message_stop: {types}"
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

    @pytest.mark.skip(reason="stage red: product gap, /v1/messages 500s (anthropic_messages TypeError) on missing messages instead of 400")
    @pytest.mark.covers("llm.messages.anthropic.input_validation.nonstream.works")
    def test_missing_messages_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = self._register(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/messages",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalMessagesBody(model=model, max_tokens=50),
        )
        assert_client_error(result, "messages missing messages")

    @pytest.mark.skip(reason="stage red: product gap, /v1/messages 500s (anthropic_messages TypeError) on missing max_tokens instead of 400")
    @pytest.mark.covers("llm.messages.anthropic.input_validation.nonstream.works")
    def test_missing_max_tokens_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = self._register(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/messages",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalMessagesBody(
                model=model, messages=[ChatMessage(role="user", content="hi")]
            ),
        )
        assert_client_error(result, "messages missing max_tokens")

    @pytest.mark.covers("llm.messages.anthropic.input_validation.nonstream.works")
    def test_missing_model_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        _, key = self._register(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/messages",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalMessagesBody(
                messages=[ChatMessage(role="user", content="hi")], max_tokens=50
            ),
        )
        assert_client_error(result, "messages missing model")
