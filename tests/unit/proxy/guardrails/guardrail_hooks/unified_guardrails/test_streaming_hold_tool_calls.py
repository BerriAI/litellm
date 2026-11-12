"""
A guardrail that scans a live stream once, at its end, can ask for tool calls to be held until that
verdict (``streaming_hold_tool_calls``). Text still streams live. From the first item that starts a
tool call the client would run, everything is held, then released on a pass or dropped on a block.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Sequence
from datetime import datetime
from typing import Final, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail, ModifyResponseException
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.chat.guardrail_translation.handler import AnthropicMessagesHandler
from litellm.llms.bedrock.passthrough.guardrail_translation.handler import (
    BedrockPassthroughGuardrailHandler,
    ConverseStreamFrame,
    InvokeStreamFrame,
    _converse_event,
    _decode_invoke_frame,
    _invoke_chunk,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import UnifiedLLMGuardrails
from litellm.responses.streaming_iterator import _CLIENT_RUN_TOOL_CALL_ITEM_TYPES
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    GenericGuardrailAPIInputs,
    ModelResponseStream,
    StreamingChoices,
    TextChoices,
    TextCompletionResponse,
)

BLOCK_MESSAGE: Final = "This tool call was blocked by policy."
TOOL_NAME: Final = "run_shell"
TOOL_ARGUMENTS: Final = '{"command": "rm -rf /"}'


def _chat_chunk(delta: Delta, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-live",
        created=1724900000,
        model="gpt-5.4-mini",
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)],
    )


def _chat_tool_delta(name: str | None, arguments: str) -> Delta:
    call: Final = ChatCompletionDeltaToolCall(
        id="call_1" if name else None, type="function", index=0, function=Function(name=name, arguments=arguments)
    )
    return Delta(tool_calls=[call])


def _chat_items(with_tool_call: bool) -> tuple[object, ...]:
    text: Final = (
        _chat_chunk(Delta(role="assistant", content="Let me ")),
        _chat_chunk(Delta(content="clean that up.")),
    )
    if not with_tool_call:
        return (*text, _chat_chunk(Delta(), finish_reason="stop"))
    return (
        *text,
        _chat_chunk(_chat_tool_delta(TOOL_NAME, "")),
        _chat_chunk(_chat_tool_delta(None, TOOL_ARGUMENTS)),
        _chat_chunk(Delta(), finish_reason="tool_calls"),
    )


def _text_completion_items(with_tool_call: bool) -> tuple[object, ...]:
    """Text completions carry no tool calls, so their translation never asks for a hold."""
    assert not with_tool_call

    def chunk(text: str, finish_reason: str | None = None) -> TextCompletionResponse:
        return TextCompletionResponse(
            id="cmpl-live",
            object="text_completion",
            created=1724900000,
            model="gpt-3.5-turbo-instruct",
            choices=[TextChoices(index=0, text=text, finish_reason=finish_reason)],
        )

    return (chunk("Let me "), chunk("clean that up."), chunk("", finish_reason="stop"))


def _sse(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _messages_events(with_tool_call: bool) -> tuple[dict[str, object], ...]:
    message: Final = {"id": "msg_live", "type": "message", "role": "assistant", "model": "claude", "content": []}
    text: Final[tuple[dict[str, object], ...]] = (
        {"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me clean that up."}},
        {"type": "content_block_stop", "index": 0},
    )
    stop_reason: Final = "tool_use" if with_tool_call else "end_turn"
    end: Final[tuple[dict[str, object], ...]] = (
        {"type": "message_delta", "delta": {"stop_reason": stop_reason}},
        {"type": "message_stop"},
    )
    if not with_tool_call:
        return (*text, *end)
    tool_block: Final = {"type": "tool_use", "id": "toolu_1", "name": TOOL_NAME, "input": {}}
    return (
        *text,
        {"type": "content_block_start", "index": 1, "content_block": tool_block},
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": TOOL_ARGUMENTS},
        },
        {"type": "content_block_stop", "index": 1},
        *end,
    )


def _messages_items(with_tool_call: bool) -> tuple[object, ...]:
    return tuple(_sse(str(event["type"]), event) for event in _messages_events(with_tool_call))


def _invoke_items(with_tool_call: bool) -> tuple[object, ...]:
    """A Bedrock invoke-with-response-stream: the ``/v1/messages`` events, one per event-stream frame."""
    return tuple(_invoke_chunk(event) for event in _messages_events(with_tool_call))


def _converse_items(with_tool_call: bool) -> tuple[object, ...]:
    text: Final = (
        _converse_event("messageStart", {"role": "assistant"}),
        _converse_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Let me clean that up."}}),
        _converse_event("contentBlockStop", {"contentBlockIndex": 0}),
    )
    if not with_tool_call:
        return (*text, _converse_event("messageStop", {"stopReason": "end_turn"}))
    return (
        *text,
        _converse_event(
            "contentBlockStart",
            {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tooluse_1", "name": TOOL_NAME}}},
        ),
        _converse_event("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": TOOL_ARGUMENTS}}}),
        _converse_event("contentBlockStop", {"contentBlockIndex": 1}),
        _converse_event("messageStop", {"stopReason": "tool_use"}),
    )


def _responses_items(with_tool_call: bool) -> tuple[object, ...]:
    envelope: Final = {"id": "resp_live", "model": "gpt-5.4-mini", "status": "in_progress", "output": []}
    message_item: Final = {"id": "msg_1", "type": "message", "role": "assistant", "content": []}
    text: Final = (
        {"type": "response.created", "response": envelope},
        {"type": "response.output_item.added", "output_index": 0, "item": message_item},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Let me clean that up.",
        },
    )
    done_message: Final = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Let me clean that up.", "annotations": []}],
    }
    call_item: Final = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": TOOL_NAME,
        "arguments": TOOL_ARGUMENTS,
        "status": "completed",
    }
    output: Final = [done_message, call_item] if with_tool_call else [done_message]
    completed: Final = {
        "type": "response.completed",
        "response": {"id": "resp_live", "model": "gpt-5.4-mini", "status": "completed", "output": output},
    }
    if not with_tool_call:
        return (*text, completed)
    return (
        *text,
        {"type": "response.output_item.added", "output_index": 1, "item": {**call_item, "arguments": ""}},
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 1,
            "delta": TOOL_ARGUMENTS,
        },
        {"type": "response.output_item.done", "output_index": 1, "item": call_item},
        completed,
    )


ROUTES: Final[dict[str, tuple[str, Callable[[bool], tuple[object, ...]], int]]] = {
    # route, the stream it relays, and how many leading items are text the client may see live
    "chat": ("/v1/chat/completions", _chat_items, 2),
    "messages": ("/v1/messages", _messages_items, 4),
    "responses": ("/v1/responses", _responses_items, 3),
    "converse": ("model/claude/converse-stream", _converse_items, 3),
    "invoke": ("model/anthropic.claude-sonnet-5-v1:0/invoke-with-response-stream", _invoke_items, 4),
}
PASS_THROUGH_ROUTES: Final = frozenset({"converse", "invoke"})
# Every way a guardrail can release a stream before its end-of-stream verdict, as the settings that select it.
STREAM_MODES: Final[dict[str, dict[str, object]]] = {
    "end_only": {"streaming_end_of_stream_only": True},
    "sampled": {"streaming_end_of_stream_only": False, "streaming_sampling_rate": 1},
    "background": {
        "streaming_end_of_stream_only": False,
        "streaming_sampling_rate": 1,
        "streaming_scan_in_background": True,
    },
    "transform_end_only": {"streaming_end_of_stream_only": True, "streaming_transform_mode": "incremental_diff"},
    "transform_sampled": {
        "streaming_end_of_stream_only": False,
        "streaming_sampling_rate": 1,
        "streaming_transform_mode": "incremental_diff",
    },
}
# The text transform runs on chat completions only.
MODES_AND_ROUTES: Final = (
    *(
        (mode, endpoint)
        for mode in ("sampled", "background")
        for endpoint in ("chat", "messages", "responses", "converse", "invoke")
    ),
    ("transform_end_only", "chat"),
    ("transform_sampled", "chat"),
)
ALL_ROUTES: Final = tuple(ROUTES)
ROUTES["text"] = ("/v1/completions", _text_completion_items, 2)


class _RecordingGuardrail(CustomGuardrail):
    """Records what the client had already received when the guardrail first scanned a tool call, or at its
    last scan when it never saw one. ``blocks_only_tool_calls`` lets the scans before the tool call pass."""

    def __init__(self, released: Sequence[object], blocks: bool, blocks_only_tool_calls: bool = False) -> None:
        super().__init__(guardrail_name="hold-tool-calls-test", event_hook="post_call")
        self._released: Final = released
        self._blocks: Final = blocks
        self._blocks_only_tool_calls: Final = blocks_only_tool_calls
        self.released_at_scan: tuple[object, ...] | None = None
        self.scanned_a_tool_call = False

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: object | None = None,
    ) -> GenericGuardrailAPIInputs:
        # Most routes hand the tool call over as a tool call; Converse hands its arguments over as a text.
        carries_tool_call: Final = _mentions_tool_call(
            [{"tool_calls": inputs.get("tool_calls"), "texts": inputs.get("texts")}]
        )
        if not self.scanned_a_tool_call:
            self.released_at_scan = tuple(self._released)
        self.scanned_a_tool_call = self.scanned_a_tool_call or carries_tool_call
        if self._blocks and (carries_tool_call or not self._blocks_only_tool_calls):
            raise ModifyResponseException(
                message=BLOCK_MESSAGE, model="m", request_data=request_data, guardrail_name=self.guardrail_name
            )
        return inputs


async def _as_stream(items: Sequence[object]) -> AsyncGenerator[object, None]:
    for item in items:
        yield item


async def _recorded(stream: AsyncIterable[object], into: list[object]) -> AsyncGenerator[object, None]:
    async for item in stream:
        into.append(item)
        yield item


async def _relay(
    endpoint: str,
    *,
    with_tool_call: bool,
    hold_tool_calls: bool,
    blocks: bool,
    wire_items: Sequence[object] | None = None,
    mode: str = "end_only",
    blocks_only_tool_calls: bool | None = None,
    scanned_a_tool_call: list[bool] | None = None,
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    """The stream's items, what the client had at scan time, and everything the client received.

    ``wire_items`` replaces the route's stream, for the same stream cut differently by the transport. ``mode`` is
    how the guardrail scans the stream, one of ``STREAM_MODES``; every mode but ``end_only`` scans more than once,
    so there the guardrail blocks only a scan that carries the tool call, unless ``blocks_only_tool_calls`` says
    otherwise. ``scanned_a_tool_call`` receives whether any scan carried the tool call."""
    route, build, _ = ROUTES[endpoint]
    wire: Final = build(with_tool_call) if wire_items is None else wire_items
    pass_through: Final = endpoint in PASS_THROUGH_ROUTES
    items: Final[list[object]] = []
    released: Final[list[object]] = []
    guardrail: Final = _RecordingGuardrail(
        released,
        blocks,
        blocks_only_tool_calls=mode != "end_only" if blocks_only_tool_calls is None else blocks_only_tool_calls,
    )
    for setting, value in STREAM_MODES[mode].items():
        setattr(guardrail, setting, value)
    guardrail.streaming_buffer_until_moderated = False
    guardrail.streaming_hold_tool_calls = hold_tool_calls
    request_data: Final[dict[str, object]] = {
        "messages": [{"role": "user", "content": "clean up"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    if pass_through:
        # A provider pass-through route maps to no call type; its logging object names it, and the proxy
        # hands the guardrail whole event-stream frames.
        request_data.update(
            custom_llm_provider="bedrock",
            endpoint=route,
            litellm_logging_obj=Logging(
                model="claude",
                messages=[],
                stream=True,
                call_type="allm_passthrough_route",
                start_time=datetime(2026, 1, 1),
                litellm_call_id="test-call",
                function_id="test-function",
            ),
        )
    upstream: Final = _as_stream(wire)
    async for chunk in UnifiedLLMGuardrails().async_post_call_streaming_iterator_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="test", request_route=None if pass_through else route),
        response=_recorded(
            BedrockPassthroughGuardrailHandler.event_stream_items(upstream, route) if pass_through else upstream,
            items,
        ),
        request_data=request_data,
    ):
        released.append(chunk)
    assert guardrail.released_at_scan is not None, "the end-of-stream scan never ran"
    if scanned_a_tool_call is not None:
        scanned_a_tool_call.append(guardrail.scanned_a_tool_call)
    return tuple(items), guardrail.released_at_scan, tuple(released)


def _readable(chunk: object) -> str:
    if isinstance(chunk, (ConverseStreamFrame, InvokeStreamFrame)):
        return repr(chunk)
    if not isinstance(chunk, bytes):
        return json.dumps(chunk, default=str)
    invoke_event: Final = _decode_invoke_frame(chunk).event
    return json.dumps(invoke_event) if invoke_event is not None else chunk.decode(errors="replace")


def _mentions_tool_call(chunks: Sequence[object]) -> bool:
    raw: Final = "".join(_readable(chunk) for chunk in chunks)
    return TOOL_NAME in raw or "rm -rf" in raw


@pytest.mark.parametrize("endpoint", ALL_ROUTES)
@pytest.mark.asyncio
async def test_a_tool_call_is_held_until_the_verdict_while_the_text_before_it_streams_live(endpoint: str) -> None:
    items, at_scan, received = await _relay(endpoint, with_tool_call=True, hold_tool_calls=True, blocks=False)
    text_items: Final = ROUTES[endpoint][2]
    assert at_scan == items[:text_items]
    assert received == items


@pytest.mark.parametrize("endpoint", ALL_ROUTES)
@pytest.mark.asyncio
async def test_a_blocked_tool_call_never_reaches_the_client(endpoint: str) -> None:
    items, at_scan, received = await _relay(endpoint, with_tool_call=True, hold_tool_calls=True, blocks=True)
    text_items: Final = ROUTES[endpoint][2]
    assert at_scan == items[:text_items]
    assert received[:text_items] == items[:text_items]
    assert not _mentions_tool_call(received)
    assert BLOCK_MESSAGE in "".join(_readable(chunk) for chunk in received)


@pytest.mark.parametrize("endpoint", ALL_ROUTES)
@pytest.mark.asyncio
async def test_without_the_flag_a_tool_call_streams_before_the_verdict(endpoint: str) -> None:
    items, at_scan, received = await _relay(endpoint, with_tool_call=True, hold_tool_calls=False, blocks=False)
    assert _mentions_tool_call(at_scan)
    assert received == items


@pytest.mark.parametrize("endpoint", ALL_ROUTES)
@pytest.mark.asyncio
async def test_a_text_only_stream_is_released_the_same_with_and_without_the_flag(endpoint: str) -> None:
    items, held_at_scan, held_received = await _relay(
        endpoint, with_tool_call=False, hold_tool_calls=True, blocks=False
    )
    _, live_at_scan, live_received = await _relay(endpoint, with_tool_call=False, hold_tool_calls=False, blocks=False)
    assert held_at_scan == live_at_scan
    assert held_received == live_received == items


@pytest.mark.asyncio
async def test_a_route_whose_stream_carries_no_tool_calls_stays_live_with_the_flag_on() -> None:
    items, at_scan, received = await _relay("text", with_tool_call=False, hold_tool_calls=True, blocks=False)
    assert at_scan == items[: ROUTES["text"][2]]
    assert received == items


def _cut(raw: bytes, size: int) -> tuple[bytes, ...]:
    return tuple(raw[start : start + size] for start in range(0, len(raw), size))


# One byte per read, reads that end inside most events, and reads that carry several events at once.
READ_SIZES: Final = (1, 7, 61, 200)


@pytest.mark.parametrize("read_size", READ_SIZES)
@pytest.mark.asyncio
async def test_a_blocked_tool_call_is_held_however_the_transport_cuts_the_messages_stream(read_size: int) -> None:
    """``/v1/messages`` relays raw transport reads, so an event can arrive split across two of them."""
    whole: Final = _messages_items(with_tool_call=True)
    text: Final = b"".join(chunk for chunk in whole[: ROUTES["messages"][2]] if isinstance(chunk, bytes))
    wire: Final = _cut(b"".join(chunk for chunk in whole if isinstance(chunk, bytes)), read_size)

    _, at_scan, received = await _relay(
        "messages", with_tool_call=True, hold_tool_calls=True, blocks=True, wire_items=wire
    )

    released_at_scan: Final = b"".join(chunk for chunk in at_scan if isinstance(chunk, bytes))
    assert text.startswith(released_at_scan)
    assert not _mentions_tool_call(received)
    assert BLOCK_MESSAGE in "".join(_readable(chunk) for chunk in received)


@pytest.mark.parametrize("read_size", READ_SIZES)
@pytest.mark.asyncio
async def test_an_allowed_messages_stream_arrives_byte_for_byte_however_the_transport_cuts_it(read_size: int) -> None:
    whole: Final = b"".join(chunk for chunk in _messages_items(with_tool_call=True) if isinstance(chunk, bytes))

    _, at_scan, received = await _relay(
        "messages", with_tool_call=True, hold_tool_calls=True, blocks=False, wire_items=_cut(whole, read_size)
    )

    assert not _mentions_tool_call(at_scan)
    assert b"".join(chunk for chunk in received if isinstance(chunk, bytes)) == whole


@pytest.mark.asyncio
async def test_text_split_across_transport_reads_still_streams_before_the_verdict() -> None:
    whole: Final = _messages_items(with_tool_call=True)
    text: Final = b"".join(chunk for chunk in whole[: ROUTES["messages"][2]] if isinstance(chunk, bytes))

    _, at_scan, _ = await _relay(
        "messages",
        with_tool_call=True,
        hold_tool_calls=True,
        blocks=False,
        wire_items=_cut(b"".join(chunk for chunk in whole if isinstance(chunk, bytes)), 1),
    )

    assert b"".join(chunk for chunk in at_scan if isinstance(chunk, bytes)) == text


def _responses_items_with_client_run_item(item_type: str) -> tuple[object, ...]:
    """A Responses stream whose tool call is an output item of ``item_type``, which has no arguments events."""
    text: Final = _responses_items(with_tool_call=False)[: ROUTES["responses"][2]]
    done_message: Final = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Let me clean that up.", "annotations": []}],
    }
    call_item: Final = {
        "id": "item_1",
        "type": item_type,
        "call_id": "call_1",
        "status": "completed",
        "action": {"type": "exec", "command": ["rm", "-rf", "/"]},
    }
    return (
        *text,
        {"type": "response.output_item.added", "output_index": 1, "item": {**call_item, "status": "in_progress"}},
        {"type": "response.output_item.done", "output_index": 1, "item": call_item},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_live",
                "model": "gpt-5.4-mini",
                "status": "completed",
                "output": [done_message, call_item],
            },
        },
    )


@pytest.mark.parametrize("item_type", sorted(_CLIENT_RUN_TOOL_CALL_ITEM_TYPES))
@pytest.mark.asyncio
async def test_every_responses_item_the_client_runs_waits_for_the_verdict_and_is_dropped_on_a_block(
    item_type: str,
) -> None:
    """The item types come from the Responses stream iterator, so a type added there is held here too.

    This shows the hold, not what the verdict reads: the guardrail here blocks the response as a whole. The
    Responses translation hands a guardrail only function and custom tool calls, so a guardrail that judges
    the other item types has to read them from the response it is given."""
    wire: Final = _responses_items_with_client_run_item(item_type)
    text_items: Final = ROUTES["responses"][2]

    items, at_scan, received = await _relay(
        "responses", with_tool_call=True, hold_tool_calls=True, blocks=True, wire_items=wire
    )

    assert at_scan == items[:text_items]
    assert received[:text_items] == items[:text_items]
    assert item_type not in "".join(_readable(chunk) for chunk in received)
    assert BLOCK_MESSAGE in "".join(_readable(chunk) for chunk in received)


@pytest.mark.parametrize(("mode", "endpoint"), MODES_AND_ROUTES)
@pytest.mark.asyncio
async def test_a_blocked_tool_call_never_reaches_the_client_in_any_mode_that_releases_early(
    mode: str, endpoint: str
) -> None:
    _, at_scan, received = await _relay(endpoint, with_tool_call=True, hold_tool_calls=True, blocks=True, mode=mode)
    assert not _mentions_tool_call(at_scan)
    assert not _mentions_tool_call(received)
    assert BLOCK_MESSAGE in "".join(_readable(chunk) for chunk in received)


@pytest.mark.parametrize(("mode", "endpoint"), MODES_AND_ROUTES)
@pytest.mark.asyncio
async def test_an_allowed_tool_call_arrives_only_after_its_scan_in_any_mode_that_releases_early(
    mode: str, endpoint: str
) -> None:
    _, at_scan, received = await _relay(endpoint, with_tool_call=True, hold_tool_calls=True, blocks=False, mode=mode)
    assert not _mentions_tool_call(at_scan)
    assert _mentions_tool_call(received)


@pytest.mark.parametrize(("mode", "endpoint"), MODES_AND_ROUTES)
@pytest.mark.asyncio
async def test_without_the_flag_those_modes_release_the_tool_call_before_its_scan(mode: str, endpoint: str) -> None:
    _, at_scan, received = await _relay(endpoint, with_tool_call=True, hold_tool_calls=False, blocks=False, mode=mode)
    assert _mentions_tool_call(at_scan)
    assert _mentions_tool_call(received)


def _without_terminal_marker(endpoint: str) -> tuple[object, ...]:
    """The route's tool-call stream as an upstream that fails or closes early leaves it: the tool call is
    complete, and the frames that end the stream never come."""
    whole: Final = ROUTES[endpoint][1](True)
    return whole[: {"chat": -1, "messages": -2, "responses": -1}[endpoint]]


UNENDED_MODES_AND_ROUTES: Final = tuple(
    (mode, endpoint) for mode, endpoint in MODES_AND_ROUTES if endpoint not in PASS_THROUGH_ROUTES
) + tuple(("end_only", endpoint) for endpoint in ("chat", "messages", "responses"))


@pytest.mark.parametrize(("mode", "endpoint"), UNENDED_MODES_AND_ROUTES)
@pytest.mark.asyncio
async def test_a_blocked_tool_call_is_dropped_when_the_stream_stops_without_its_terminal_marker(
    mode: str, endpoint: str
) -> None:
    guardrail_saw: Final[list[bool]] = []

    _, at_scan, received = await _relay(
        endpoint,
        with_tool_call=True,
        hold_tool_calls=True,
        blocks=True,
        mode=mode,
        wire_items=_without_terminal_marker(endpoint),
        blocks_only_tool_calls=True,
        scanned_a_tool_call=guardrail_saw,
    )

    assert guardrail_saw == [True], "the verdict that released or dropped the tool call never covered it"
    assert not _mentions_tool_call(at_scan)
    assert not _mentions_tool_call(received)
    assert BLOCK_MESSAGE in "".join(_readable(chunk) for chunk in received)


@pytest.mark.parametrize(("mode", "endpoint"), UNENDED_MODES_AND_ROUTES)
@pytest.mark.asyncio
async def test_an_allowed_tool_call_is_scanned_before_release_when_the_stream_stops_without_its_terminal_marker(
    mode: str, endpoint: str
) -> None:
    guardrail_saw: Final[list[bool]] = []

    _, at_scan, received = await _relay(
        endpoint,
        with_tool_call=True,
        hold_tool_calls=True,
        blocks=False,
        mode=mode,
        wire_items=_without_terminal_marker(endpoint),
        scanned_a_tool_call=guardrail_saw,
    )

    assert guardrail_saw == [True], "the tool call was released on a verdict that never covered it"
    assert not _mentions_tool_call(at_scan)
    assert _mentions_tool_call(received)


@pytest.mark.parametrize("mode", ["sampled", "background"])
@pytest.mark.asyncio
async def test_a_tool_call_that_only_the_final_responses_frame_carries_waits_for_the_verdict(mode: str) -> None:
    whole: Final = _responses_items(with_tool_call=True)
    wire: Final = (*whole[: ROUTES["responses"][2]], whole[-1])

    _, at_scan, received = await _relay(
        "responses", with_tool_call=True, hold_tool_calls=True, blocks=True, mode=mode, wire_items=wire
    )

    assert not _mentions_tool_call(at_scan)
    assert not _mentions_tool_call(received)


@pytest.mark.parametrize("blocks", [True, False])
@pytest.mark.asyncio
async def test_a_messages_stream_cut_off_inside_its_last_frame_keeps_the_hold(blocks: bool) -> None:
    whole: Final = b"".join(chunk for chunk in _messages_items(with_tool_call=True) if isinstance(chunk, bytes))
    cut_off: Final = whole[:-5]

    _, at_scan, received = await _relay(
        "messages", with_tool_call=True, hold_tool_calls=True, blocks=blocks, wire_items=_cut(cut_off, 61)
    )

    assert not _mentions_tool_call(at_scan)
    if blocks:
        assert not _mentions_tool_call(received)
    else:
        assert b"".join(chunk for chunk in received if isinstance(chunk, bytes)) == cut_off


def test_the_messages_translation_cuts_a_transport_read_at_its_last_whole_frame() -> None:
    first, second = _messages_items(with_tool_call=True)[:2]
    assert isinstance(first, bytes) and isinstance(second, bytes)
    translation: Final = AnthropicMessagesHandler()

    assert translation.split_whole_stream_frames(None, first) == (first, None)
    assert translation.split_whole_stream_frames(None, first + second[:9]) == (first, second[:9])
    assert translation.split_whole_stream_frames(None, second[:9]) == (None, second[:9])
    assert translation.split_whole_stream_frames(second[:9], second[9:]) == (second, None)
    # An event some providers hand over already parsed is a whole frame.
    parsed: Final = {"type": "message_stop"}
    assert translation.split_whole_stream_frames(None, parsed) == (parsed, None)
