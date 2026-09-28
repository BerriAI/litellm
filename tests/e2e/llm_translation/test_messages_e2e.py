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
from collections.abc import Callable
from types import MappingProxyType
from typing import Final

import anthropic
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
from e2e_config import (
    PROVIDER_EDGE_ADVERTISE_HOST,
    PROVIDER_EDGE_BIND_HOST,
    STREAM_MIN_LEAD_SECONDS,
    provider_edge_base,
    provider_paces_stream,
    unique_marker,
)
from e2e_http import assert_client_error
from lifecycle import ResourceManager
from models import AnthropicErrorEvent, AnthropicMessagesBody, ChatMessage, LiteLLMParamsBody, SpendLogRow
from provider_edge import EDGE_MOUNTS, LiveEdge, RunningEdge, StreamCut, start_provider_edge
from provider_edge_bedrock import bedrock_signer
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
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


BEDROCK_BACKEND: Final = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_EDGE_REGION: Final = "us-east-1"
_STREAM_FAILURE_PROMPT: Final = "Count from 1 to 100, one number per line."
_FRAME_PAYLOAD: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_AT_FRAME_BOUNDARY: Final = StreamCut(after_content=True)
_MID_FRAME: Final = StreamCut(after_content=True, mid_chunk=True)
_BEFORE_FIRST_BYTE: Final = StreamCut(after_content=False)

type _CutRegistration = Callable[[ProxyClient, ResourceManager, StreamCut], tuple[str, str]]


def _cut_edge(backend: LiveEdge, mount: str) -> RunningEdge:
    return start_provider_edge(
        backend,
        mounts=MappingProxyType({mount: EDGE_MOUNTS[mount]}),
        bind_host=PROVIDER_EDGE_BIND_HOST,
        advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
    )


def _register_cut_bedrock(proxy: ProxyClient, resources: ResourceManager, cut: StreamCut) -> tuple[str, str]:
    mount: Final = f"bedrock/{BEDROCK_EDGE_REGION}"
    edge: Final = _cut_edge(LiveEdge(cut=cut, sign=bedrock_signer(BEDROCK_EDGE_REGION)), mount)
    resources.defer(edge.shutdown)
    return _register(
        proxy,
        resources,
        LiteLLMParamsBody(
            model=BEDROCK_BACKEND,
            api_base=edge.edge.api_base(mount),
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name=BEDROCK_EDGE_REGION,
        ),
        prefix="e2e-messages-cut",
    )


def _register_cut_anthropic(proxy: ProxyClient, resources: ResourceManager, cut: StreamCut) -> tuple[str, str]:
    edge: Final = _cut_edge(LiveEdge(cut=cut), "anthropic")
    resources.defer(edge.shutdown)
    return _register(
        proxy,
        resources,
        LiteLLMParamsBody(
            model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY", api_base=edge.edge.api_base("anthropic")
        ),
        prefix="e2e-messages-cut",
    )


_DROPPED_UPSTREAMS: Final[tuple[tuple[str, _CutRegistration, StreamCut], ...]] = (
    ("bedrock_at_a_frame_boundary", _register_cut_bedrock, _AT_FRAME_BOUNDARY),
    ("anthropic_at_a_frame_boundary", _register_cut_anthropic, _AT_FRAME_BOUNDARY),
    ("anthropic_mid_frame", _register_cut_anthropic, _MID_FRAME),
)
_DROPPED_BEFORE_FIRST_BYTE: Final[tuple[tuple[str, _CutRegistration, StreamCut], ...]] = (
    ("bedrock_before_the_first_byte", _register_cut_bedrock, _BEFORE_FIRST_BYTE),
    ("anthropic_before_the_first_byte", _register_cut_anthropic, _BEFORE_FIRST_BYTE),
)


def _payload(frame: str) -> JsonValue | None:
    try:
        return _FRAME_PAYLOAD.validate_json(frame)
    except ValidationError:
        return None


def _bare_error_frame(frame: str) -> bool:
    payload: Final = _payload(frame)
    return isinstance(payload, dict) and "error" in payload and payload.get("type") != "error"


@pytest.mark.provider_edge_host
@pytest.mark.provider_live
class TestMessagesUpstreamStreamFailure:
    @pytest.mark.covers("llm.messages.anthropic.upstream_stream_failure.stream.error_event")
    @pytest.mark.parametrize(
        ("register", "cut"), [case[1:] for case in _DROPPED_UPSTREAMS], ids=[case[0] for case in _DROPPED_UPSTREAMS]
    )
    def test_interrupted_upstream_stream_raises_in_the_anthropic_sdk(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        register: _CutRegistration,
        cut: StreamCut,
    ) -> None:
        model, key = register(proxy, resources, cut)
        client: Final = sdk.anthropic(key)

        stream: Final = client.messages.create(
            model=model,
            max_tokens=300,
            stream=True,
            messages=[_user_turn(_STREAM_FAILURE_PROMPT)],
            extra_body=NO_PROXY_CACHE,
        )
        first: Final = next(stream)
        assert first.type == "message_start", (
            f"the stream produced a first event that is not message_start, so this run proves a "
            f"startup failure, not an interrupted stream: {first!r}"
        )
        with pytest.raises(anthropic.APIStatusError) as raised:
            for _ in stream:
                pass
        try:
            AnthropicErrorEvent.model_validate(raised.value.body)
        except ValidationError:
            pytest.fail(
                f"the SDK raised on the interrupted stream but without the Anthropic error envelope a "
                f"client reads the failure from: body={raised.value.body!r} message={raised.value}"
            )

    @pytest.mark.covers("llm.messages.anthropic.upstream_stream_failure.stream.error_event")
    @pytest.mark.parametrize(
        ("register", "cut"), [case[1:] for case in _DROPPED_UPSTREAMS], ids=[case[0] for case in _DROPPED_UPSTREAMS]
    )
    def test_interrupted_upstream_stream_is_an_anthropic_error_event(
        self, proxy: ProxyClient, resources: ResourceManager, register: _CutRegistration, cut: StreamCut
    ) -> None:
        model, key = register(proxy, resources, cut)

        outcome: Final = proxy.messages_stream(
            key,
            AnthropicMessagesBody(
                model=model,
                max_tokens=300,
                stream=True,
                messages=[ChatMessage(role="user", content=_STREAM_FAILURE_PROMPT)],
            ),
        )
        frames: Final = outcome.stream_events
        assert outcome.is_streaming, (
            f"/v1/messages did not answer with an SSE stream: status={outcome.status_code} body={outcome.body}"
        )
        assert frames, (
            f"the proxy sent no SSE data frames although the upstream hung up; stream_error={outcome.stream_error!r}"
        )
        assert outcome.stream_error == "event: error", (
            f"the interrupted stream was not announced by an 'event: error' line Anthropic clients read; "
            f"stream_error={outcome.stream_error!r} frames={frames}"
        )
        try:
            AnthropicErrorEvent.model_validate_json(frames[-1])
        except ValidationError:
            pytest.fail(
                f'the last SSE frame was not an Anthropic {{"type": "error", "error": ...}} envelope; frames={frames}'
            )
        torn: Final = tuple(index for index, frame in enumerate(frames) if _payload(frame) is None)
        expected_torn: Final = 1 if cut.mid_chunk else 0
        assert len(torn) == expected_torn, (
            f"expected {expected_torn} data line(s) that are not JSON, since the edge tears one only when it "
            f"cuts mid-frame, but the proxy relayed {[frames[index] for index in torn]}; all frames={frames}"
        )
        for index in torn:
            assert _payload(frames[index + 1]) == {"type": "ping"}, (
                f"the frame the upstream tore was not closed as a ping event before the error, so an "
                f"Anthropic client parses the error inside it: after {frames[index]!r} came "
                f"{frames[index + 1]!r}; all frames={frames}"
            )
        bare: Final = tuple(frame for frame in frames if _bare_error_frame(frame))
        assert not bare, (
            f"the proxy emitted error frames without the Anthropic envelope, which Anthropic clients drop: "
            f"{bare}; all frames={frames}"
        )

    @pytest.mark.covers("llm.messages.anthropic.upstream_stream_failure.stream.error_status")
    @pytest.mark.parametrize(
        ("register", "cut"),
        [case[1:] for case in _DROPPED_BEFORE_FIRST_BYTE],
        ids=[case[0] for case in _DROPPED_BEFORE_FIRST_BYTE],
    )
    def test_upstream_that_hangs_up_before_the_first_byte_raises_with_its_status_in_the_anthropic_sdk(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        register: _CutRegistration,
        cut: StreamCut,
    ) -> None:
        model, key = register(proxy, resources, cut)
        client: Final = sdk.anthropic(key)

        with pytest.raises(anthropic.APIStatusError) as raised:
            client.messages.create(
                model=model,
                max_tokens=300,
                stream=True,
                messages=[_user_turn(_STREAM_FAILURE_PROMPT)],
                extra_body=NO_PROXY_CACHE,
            )
        assert 500 <= raised.value.status_code < 600, (
            f"an upstream that hung up before sending anything must answer with a server error status the SDK "
            f"retries on, not {raised.value.status_code}: {raised.value}"
        )
        try:
            AnthropicErrorEvent.model_validate(raised.value.body)
        except ValidationError:
            pytest.fail(
                f"the SDK raised with the right status but without the Anthropic error envelope a client reads "
                f"the failure from: body={raised.value.body!r} message={raised.value}"
            )

    @pytest.mark.covers("llm.messages.anthropic.upstream_stream_failure.stream.error_status")
    @pytest.mark.parametrize(
        ("register", "cut"),
        [case[1:] for case in _DROPPED_BEFORE_FIRST_BYTE],
        ids=[case[0] for case in _DROPPED_BEFORE_FIRST_BYTE],
    )
    def test_upstream_that_hangs_up_before_the_first_byte_is_a_json_error_with_its_status(
        self, proxy: ProxyClient, resources: ResourceManager, register: _CutRegistration, cut: StreamCut
    ) -> None:
        model, key = register(proxy, resources, cut)

        outcome: Final = proxy.messages_stream(
            key,
            AnthropicMessagesBody(
                model=model,
                max_tokens=300,
                stream=True,
                messages=[ChatMessage(role="user", content=_STREAM_FAILURE_PROMPT)],
            ),
        )
        assert not outcome.is_streaming, (
            f"nothing had been streamed when the upstream hung up, yet /v1/messages opened a 200 SSE stream "
            f"instead of answering with the failure's status: stream_error={outcome.stream_error!r} "
            f"frames={outcome.stream_events}"
        )
        assert 500 <= outcome.status_code < 600, (
            f"/v1/messages answered {outcome.status_code} for an upstream that hung up before its first byte; "
            f"body={outcome.body}"
        )
        try:
            AnthropicErrorEvent.model_validate_json(outcome.body)
        except ValidationError:
            pytest.fail(
                f'the error body is not an Anthropic {{"type": "error", "error": ...}} envelope; body={outcome.body}'
            )
