"""Live e2e: POST /v1/messages (Anthropic Messages API) returns a real completion.

Registers an Anthropic deployment at runtime, drives the Messages endpoint through
the gateway with the real Anthropic SDK, the client customers actually use
(LIT-4577), and asserts an assistant message with text came back, both
non-streaming and streamed. Malformed bodies the SDK refuses to build stay on the
shared transport. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import time
from typing import Final

import pytest
from anthropic import Anthropic
from anthropic.types import (
    InputJSONDelta,
    Message,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStreamEvent,
    TextBlock,
    TextDelta,
    ToolChoiceParam,
    ToolParam,
    ToolUseBlock,
)
from e2e_config import STREAM_MIN_LEAD_SECONDS, provider_edge_base, provider_paces_stream, unique_marker
from e2e_http import assert_client_error
from lifecycle import ResourceManager
from models import ChatMessage, LiteLLMParamsBody, SpendLogRow
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict
from sdk_clients import NO_PROXY_CACHE, SdkClients, response_header

pytestmark = [pytest.mark.e2e, pytest.mark.replayable]


class _OptionalMessagesBody(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int | None = None


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


def _anthropic_params() -> LiteLLMParamsBody:
    """The Anthropic deployment, wired through the record/replay edge when a fixture
    mode is active (LIT-5974). The mount base carries no ``/v1``: litellm's Anthropic
    handler appends ``/v1/messages`` to ``api_base`` itself, where the OpenAI handler
    appends only ``/chat/completions``."""
    base = provider_edge_base("anthropic")
    return LiteLLMParamsBody(model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY", api_base=base)


def _register(
    proxy: ProxyClient,
    resources: ResourceManager,
    params: LiteLLMParamsBody | None = None,
    prefix: str = "e2e-messages",
) -> tuple[str, str]:
    model = f"{prefix}-{unique_marker()}"
    model_id = proxy.create_model(model, _anthropic_params() if params is None else params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _text(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def _user_turn(text: str) -> MessageParam:
    return {"role": "user", "content": text}


class TestAnthropicMessages:
    @pytest.mark.covers("llm.messages.anthropic.basic.nonstream.works")
    def test_messages_returns_completion(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _register(proxy, resources)
        client = sdk.anthropic(key)

        message = client.messages.create(
            model=model, max_tokens=64, messages=[_user_turn("reply with one word")], extra_body=NO_PROXY_CACHE
        )
        assert message.role == "assistant", f"unexpected role: {message.role!r}"
        assert _text(message).strip(), f"/v1/messages returned no text: {message.content!r}"

    @pytest.mark.covers("llm.messages.anthropic.basic.nonstream.cost_logged")
    def test_messages_logs_cost_matching_the_response_header(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, prefix="e2e-messages-cost")
        client = sdk.anthropic(key)

        raw = client.messages.with_raw_response.create(
            model=model,
            max_tokens=64,
            messages=[_user_turn(f"reply with one word {unique_marker()}")],
            extra_body=NO_PROXY_CACHE,
        )
        message = raw.parse()
        assert message.role == "assistant" and _text(message).strip(), (
            f"/v1/messages returned no assistant text: {message.content!r}"
        )

        # The customer reads per-request cost off the response header (LIT-4076), so
        # it must be present and positive on /v1/messages, not only /chat/completions.
        raw_header_cost = response_header(raw.headers, "x-litellm-response-cost")
        assert raw_header_cost is not None, (
            f"x-litellm-response-cost header missing on /v1/messages; headers={dict(raw.headers)}"
        )
        header_cost = float(raw_header_cost)
        assert header_cost > 0, f"x-litellm-response-cost header non-positive on /v1/messages: {header_cost}"

        # Correlate the spend row by the unique scoped key, not the Anthropic response
        # id: on /v1/messages the spend-log request_id is the proxy's own call id, which
        # need not equal the message body id, so an id-based poll can miss a correctly
        # logged row and time out. The key is fresh per test, so its only priced row is
        # this call.
        def _priced(rows: list[SpendLogRow]) -> bool:
            return any(r.spend is not None and r.spend > 0 for r in rows)

        rows = proxy.poll_logs_for_key(key, predicate=_priced)
        priced = [r for r in rows if r.spend is not None and r.spend > 0]
        assert priced, f"no priced /spend/logs row landed for key {key} within the poll window; got {rows}"
        row = priced[0]
        assert (row.prompt_tokens or 0) > 0 and (row.completion_tokens or 0) > 0, (
            f"messages spend row missing token counts, so the cost is not real usage: {row}"
        )
        assert row.spend is not None and _approx_equal(row.spend, header_cost), (
            f"logged spend {row.spend} disagrees with the x-litellm-response-cost header {header_cost}; "
            "the customer bills against the header, so the two must match"
        )

    @pytest.mark.covers("llm.messages.anthropic.basic.stream.works")
    @pytest.mark.provider_live
    def test_messages_streams_completion(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        """Edge-wired like its non-streaming siblings, so record and replay both
        carry the streamed response.

        Asserts what the proxy controls: the event grammar (usage between the last
        content delta and ``message_stop``) and, on the clock, that the relay is
        incremental. How many deltas a reply is split into is the provider's choice, so
        the first content delta must instead reach the client well before
        ``message_stop``, which a buffered response cannot do. Replay serves chunks back
        to back, so only live and record runs judge the timing."""
        model, key = _register(proxy, resources)
        client = sdk.anthropic(key)

        started: Final = time.monotonic()
        stream = client.messages.create(
            model=model,
            max_tokens=800,
            stream=True,
            messages=[_user_turn("Count from 1 to 200, one number per line.")],
            extra_body=NO_PROXY_CACHE,
        )
        arrivals: Final = tuple((event, time.monotonic() - started) for event in stream)
        assert arrivals, "stream produced no SSE events"

        events: Final = tuple(event for event, _ in arrivals)
        types: Final = tuple(event.type for event in events)
        delta_positions: Final = tuple(
            index for index, event in enumerate(events) if event.type == "content_block_delta"
        )
        assert delta_positions, f"stream carried no content deltas: {types}"
        text: Final = "".join(
            event.delta.text
            for event in events
            if isinstance(event, RawContentBlockDeltaEvent) and isinstance(event.delta, TextDelta)
        )
        assert text.strip(), f"content deltas assembled to no text: {events[:5]}"

        usage_positions: Final = tuple(
            index for index, event in enumerate(events) if isinstance(event, RawMessageDeltaEvent)
        )
        assert usage_positions, f"stream never reported usage: {types}"
        assert "message_stop" in types, f"stream never reached message_stop: {types}"
        stop_position: Final = types.index("message_stop")
        assert delta_positions[-1] < usage_positions[0] < stop_position, (
            f"usage did not land between the last content delta and message_stop: {types}"
        )

        first_delta_at: Final = arrivals[delta_positions[0]][1]
        stop_at: Final = arrivals[stop_position][1]
        if provider_paces_stream():
            assert stop_at - first_delta_at >= STREAM_MIN_LEAD_SECONDS, (
                f"first content delta reached the client {first_delta_at:.2f}s after the request "
                f"and message_stop {stop_at:.2f}s after it; a relayed stream shows the first delta "
                f"at least {STREAM_MIN_LEAD_SECONDS}s before the end, so the response was buffered"
            )

    @pytest.mark.covers("llm.messages.anthropic.tool_use.nonstream.works")
    def test_messages_tool_use(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _register(proxy, resources)
        client = sdk.anthropic(key)

        message = client.messages.create(
            model=model,
            max_tokens=256,
            tools=[WEATHER_TOOL],
            messages=[_user_turn("What is the weather in Paris? Use the tool.")],
            extra_body=NO_PROXY_CACHE,
        )
        assert message.content, f"no content blocks in response: {message!r}"
        assert any(isinstance(block, ToolUseBlock) for block in message.content), (
            f"model did not call the tool: {message.content!r}"
        )

    @pytest.mark.skip(
        reason="stage red: product gap, /v1/messages 500s (anthropic_messages TypeError) on missing messages instead of 400"
    )
    @pytest.mark.covers("llm.messages.anthropic.input_validation.nonstream.works")
    def test_missing_messages_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            "/v1/messages",
            headers=proxy.transport.bearer(key),
            json=_OptionalMessagesBody(model=model, max_tokens=50),
        )
        assert_client_error(result, "messages missing messages")

    @pytest.mark.skip(
        reason="stage red: product gap, /v1/messages 500s (anthropic_messages TypeError) on missing max_tokens instead of 400"
    )
    @pytest.mark.covers("llm.messages.anthropic.input_validation.nonstream.works")
    def test_missing_max_tokens_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            "/v1/messages",
            headers=proxy.transport.bearer(key),
            json=_OptionalMessagesBody(model=model, messages=[ChatMessage(role="user", content="hi")]),
        )
        assert_client_error(result, "messages missing max_tokens")

    @pytest.mark.covers("llm.messages.anthropic.input_validation.nonstream.works")
    def test_missing_model_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        _, key = _register(proxy, resources)
        result = proxy.transport.send(
            "/v1/messages",
            headers=proxy.transport.bearer(key),
            json=_OptionalMessagesBody(messages=[ChatMessage(role="user", content="hi")], max_tokens=50),
        )
        assert_client_error(result, "messages missing model")


class _ParcelInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    parcel: str
    shelf: int


def _tool_from_stream(events: tuple[RawMessageStreamEvent, ...]) -> ToolUseBlock:
    starts: Final = tuple(
        (index, event.index, event.content_block)
        for index, event in enumerate(events)
        if isinstance(event, RawContentBlockStartEvent) and isinstance(event.content_block, ToolUseBlock)
    )
    assert len(starts) == 1, "expected exactly one tool call"
    start_position, block_index, block = starts[0]
    assert block.id
    fragments: Final = tuple(
        (index, event.index, event.delta.partial_json)
        for index, event in enumerate(events)
        if isinstance(event, RawContentBlockDeltaEvent) and isinstance(event.delta, InputJSONDelta)
    )
    assert fragments, "tool stream contained no argument fragments"
    assert all(fragment_block == block_index for _, fragment_block, _ in fragments), "tool fragments changed index"
    positions: Final = tuple(index for index, _, _ in fragments)
    stops: Final = tuple(
        index
        for index, event in enumerate(events)
        if isinstance(event, RawContentBlockStopEvent) and event.index == block_index
    )
    assert len(stops) == 1 and start_position < positions[0] <= positions[-1] < stops[0]
    terminal_positions: Final = tuple(
        index for index, event in enumerate(events) if isinstance(event, RawMessageDeltaEvent)
    )
    stop_reasons: Final = tuple(event.delta.stop_reason for event in events if isinstance(event, RawMessageDeltaEvent))
    assert stop_reasons == ("tool_use",)
    assert len(terminal_positions) == 1 and stops[0] < terminal_positions[0] < len(events) - 1
    assert tuple(index for index, event in enumerate(events) if event.type == "message_stop") == (len(events) - 1,), (
        "tool stream did not terminate exactly once"
    )
    arguments: Final = _ParcelInput.model_validate_json("".join(partial for _, _, partial in fragments))
    return ToolUseBlock(type="tool_use", id=block.id, name=block.name, input=arguments.model_dump())


def _request_tool(client: Anthropic, model: str, question: MessageParam, tool: ToolParam, stream: bool) -> ToolUseBlock:
    tool_choice: Final[ToolChoiceParam] = {"type": "tool", "name": tool["name"]}
    if stream:
        events: Final = tuple(
            client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[question],
                tools=[tool],
                tool_choice=tool_choice,
                stream=True,
                extra_body=NO_PROXY_CACHE,
            )
        )
        return _tool_from_stream(events)
    message: Final = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[question],
        tools=[tool],
        tool_choice=tool_choice,
        extra_body=NO_PROXY_CACHE,
    )
    blocks: Final = tuple(block for block in message.content if isinstance(block, ToolUseBlock))
    assert len(blocks) == 1
    return blocks[0]


class TestOpenAIMessagesToolContinuation:
    @pytest.mark.provider_live
    @pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
    def test_required_tool_arguments_and_correlated_result(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, stream: bool
    ) -> None:
        model: Final = f"e2e-bridge-tool-{unique_marker()}"
        base: Final = provider_edge_base("openai")
        model_id: Final = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-5.6", api_key="os.environ/OPENAI_API_KEY", api_base=f"{base}/v1" if base else None
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        client: Final = sdk.anthropic(resources.key(models=[model]))
        tool: Final[ToolParam] = {
            "name": "locate_parcel",
            "description": "Look up the receipt for a parcel on a shelf. Return the receipt verbatim.",
            "input_schema": {
                "type": "object",
                "properties": {"parcel": {"type": "string"}, "shelf": {"type": "integer"}},
                "required": ["parcel", "shelf"],
            },
        }
        question: Final = _user_turn(
            "Call locate_parcel with parcel exactly amber-kite and shelf exactly 7. "
            "After the tool result, reply with only the receipt returned by the tool."
        )
        emitted: Final = _request_tool(client, model, question, tool, stream)
        assert emitted.id and emitted.name == "locate_parcel"
        assert emitted.input == {"parcel": "amber-kite", "shelf": 7}, "required tool arguments were lost or changed"
        receipt: Final = f"receipt-{unique_marker()}"
        continuation: Final = client.messages.create(
            model=model,
            max_tokens=2048,
            tools=[tool],
            tool_choice={"type": "none"},
            messages=[
                question,
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": emitted.id, "name": emitted.name, "input": emitted.input}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": emitted.id, "content": receipt}]},
            ],
            extra_body=NO_PROXY_CACHE,
        )
        assert _text(continuation).strip() == receipt, "continuation did not consume the correlated tool result"
        assert all(not isinstance(block, ToolUseBlock) for block in continuation.content)
