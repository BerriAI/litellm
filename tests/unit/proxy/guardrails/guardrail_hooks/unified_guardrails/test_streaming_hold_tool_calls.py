"""
A guardrail that scans a live stream once, at its end, can ask for tool calls to be held until that
verdict (``streaming_hold_tool_calls``). Text still streams live. From the first item that starts a
tool call the client would run, everything is held, then released on a pass or dropped on a block.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail, ModifyResponseException
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.a2a.chat.guardrail_translation.handler import A2AGuardrailHandler
from litellm.llms.anthropic.chat.guardrail_translation.handler import AnthropicMessagesHandler
from litellm.llms.bedrock.passthrough.guardrail_translation.handler import (
    BedrockPassthroughGuardrailHandler,
    ConverseStreamFrame,
    InvokeStreamFrame,
    _converse_event,
    _decode_invoke_frame,
    _invoke_chunk,
)
from litellm.llms.gemini.google_genai.guardrail_translation.handler import GoogleGenAIGenerateContentHandler
from litellm.llms.openai.image_generation.guardrail_translation.handler import OpenAIImageGenerationHandler
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


def _gemini_candidate(part: dict[str, object], finish_reason: str | None = None) -> dict[str, object]:
    candidate: Final[dict[str, object]] = {"content": {"role": "model", "parts": [part]}, "index": 0}
    return candidate if finish_reason is None else {**candidate, "finishReason": finish_reason}


def _gemini_chunk(part: dict[str, object], finish_reason: str | None = None) -> dict[str, object]:
    return {"candidates": [_gemini_candidate(part, finish_reason)]}


GEMINI_FUNCTION_CALL: Final[dict[str, object]] = {
    "functionCall": {"name": TOOL_NAME, "args": json.loads(TOOL_ARGUMENTS)}
}


def _gemini_items(with_tool_call: bool) -> tuple[object, ...]:
    """A native Gemini ``:streamGenerateContent`` stream, as the proxy hands it over: one SSE event per item."""

    def event(chunk: dict[str, object]) -> bytes:
        return f"data: {json.dumps(chunk)}\n\n".encode()

    text: Final = (event(_gemini_chunk({"text": "Let me "})), event(_gemini_chunk({"text": "clean that up."})))
    if not with_tool_call:
        return (*text, event(_gemini_chunk({"text": ""}, finish_reason="STOP")))
    return (*text, event(_gemini_chunk(GEMINI_FUNCTION_CALL, finish_reason="STOP")))


def _a2a_items(with_tool_call: bool) -> tuple[object, ...]:
    """An A2A ``message/stream`` stream. Its translation scans the stream but does not tell a tool call apart."""

    def update(text: str) -> dict[str, object]:
        artifact: Final = {"artifactId": "a1", "parts": [{"kind": "text", "text": text}]}
        return {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"kind": "artifact-update", "taskId": "t1", "artifact": artifact},
        }

    status: Final = {"kind": "status-update", "taskId": "t1", "status": {"state": "completed"}, "final": True}
    return (update("Let me "), update("clean that up."), {"jsonrpc": "2.0", "id": "1", "result": status})


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
ROUTES["gemini"] = ("/v1beta/models/gemini-2.5-flash:streamGenerateContent", _gemini_items, 2)
ROUTES["a2a"] = ("/a2a/cleanup-agent/message/stream", _a2a_items, 0)


class _RecordingGuardrail(CustomGuardrail):
    """Records what the client had already received when the guardrail first scanned a tool call, or at its
    last scan when it never saw one. ``blocks_only_tool_calls`` lets the scans before the tool call pass."""

    def __init__(
        self,
        released: Sequence[object],
        blocks: bool,
        blocks_only_tool_calls: bool = False,
        reads_response_for: str | None = None,
    ) -> None:
        super().__init__(guardrail_name="hold-tool-calls-test", event_hook="post_call")
        self._released: Final = released
        self._blocks: Final = blocks
        self._blocks_only_tool_calls: Final = blocks_only_tool_calls
        # An output item type this guardrail looks for in the whole response it is given, the way a guardrail
        # that does not rely on the translation's inputs judges an item those inputs leave out.
        self._reads_response_for: Final = reads_response_for
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
        ) or (
            self._reads_response_for is not None
            and self._reads_response_for
            in json.dumps(request_data.get("response"), default=str) + _stream_text(request_data)
        )
        if not self.scanned_a_tool_call:
            self.released_at_scan = tuple(self._released)
        self.scanned_a_tool_call = self.scanned_a_tool_call or carries_tool_call
        if self._blocks and (carries_tool_call or not self._blocks_only_tool_calls):
            raise ModifyResponseException(
                message=BLOCK_MESSAGE, model="m", request_data=request_data, guardrail_name=self.guardrail_name
            )
        return inputs


def _stream_text(request_data: dict) -> str:
    """The raw stream a translation hands over as ``responses_so_far``, joined, so a read cut anywhere reads whole."""
    stream: Final = request_data.get("responses_so_far")
    if not isinstance(stream, (list, tuple)):
        return ""
    return b"".join(item for item in stream if isinstance(item, bytes)).decode(errors="replace")


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
    reads_response_for: str | None = None,
    released_into: list[object] | None = None,
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    """The stream's items, what the client had at scan time, and everything the client received.

    ``wire_items`` replaces the route's stream, for the same stream cut differently by the transport. ``mode`` is
    how the guardrail scans the stream, one of ``STREAM_MODES``; every mode but ``end_only`` scans more than once,
    so there the guardrail blocks only a scan that carries the tool call, unless ``blocks_only_tool_calls`` says
    otherwise. ``scanned_a_tool_call`` receives whether any scan carried the tool call. ``released_into`` receives
    what the client was sent, for a route that ends a blocked stream by raising."""
    route, build, _ = ROUTES[endpoint]
    wire: Final = build(with_tool_call) if wire_items is None else wire_items
    pass_through: Final = endpoint in PASS_THROUGH_ROUTES
    items: Final[list[object]] = []
    released: Final[list[object]] = [] if released_into is None else released_into
    guardrail: Final = _RecordingGuardrail(
        released,
        blocks,
        blocks_only_tool_calls=mode != "end_only" if blocks_only_tool_calls is None else blocks_only_tool_calls,
        reads_response_for=reads_response_for,
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


@pytest.mark.parametrize("mode", ["end_only", "sampled", "background"])
@pytest.mark.parametrize("item_type", sorted(_CLIENT_RUN_TOOL_CALL_ITEM_TYPES - {"function_call", "custom_tool_call"}))
@pytest.mark.asyncio
async def test_a_held_item_the_scan_key_does_not_carry_still_gets_an_end_of_stream_round(
    mode: str, item_type: str
) -> None:
    """The Responses scan key carries only function and custom tool calls, so with one of the other items held
    the end key equals the key of the round that scanned the text before it. That round must not make the
    end-of-stream round redundant: a guardrail that reads the whole response judges the item there."""
    guardrail_saw: Final[list[bool]] = []

    _, _, received = await _relay(
        "responses",
        with_tool_call=True,
        hold_tool_calls=True,
        blocks=True,
        mode=mode,
        wire_items=_responses_items_with_client_run_item(item_type),
        blocks_only_tool_calls=True,
        scanned_a_tool_call=guardrail_saw,
        reads_response_for=item_type,
    )

    assert guardrail_saw == [True], "the held item was released with no round after it arrived"
    assert item_type not in "".join(_readable(chunk) for chunk in received)
    assert BLOCK_MESSAGE in "".join(_readable(chunk) for chunk in received)


EARLY_RELEASE_MODES: Final = pytest.mark.parametrize("mode", ["end_only", "sampled", "background"])


@EARLY_RELEASE_MODES
@pytest.mark.asyncio
async def test_a_gemini_function_call_is_held_until_the_verdict_while_the_text_before_it_streams_live(
    mode: str,
) -> None:
    items, at_scan, received = await _relay(
        "gemini", with_tool_call=True, hold_tool_calls=True, blocks=False, mode=mode, blocks_only_tool_calls=False
    )

    assert at_scan == items[:2]
    assert received == items


@EARLY_RELEASE_MODES
@pytest.mark.asyncio
async def test_a_blocked_gemini_function_call_never_reaches_the_client(mode: str) -> None:
    """This route has no block frames of its own, so a block ends the stream by raising, as it does today."""
    released: Final[list[object]] = []

    with pytest.raises(ModifyResponseException):
        await _relay(
            "gemini",
            with_tool_call=True,
            hold_tool_calls=True,
            blocks=True,
            mode=mode,
            blocks_only_tool_calls=True,
            reads_response_for="functionCall",
            released_into=released,
        )

    assert released == list(_gemini_items(with_tool_call=True)[:2])
    assert not _mentions_tool_call(released)


@pytest.mark.asyncio
async def test_without_the_flag_a_blocked_gemini_function_call_reaches_the_client_first() -> None:
    released: Final[list[object]] = []

    with pytest.raises(ModifyResponseException):
        await _relay(
            "gemini",
            with_tool_call=True,
            hold_tool_calls=False,
            blocks=True,
            blocks_only_tool_calls=True,
            reads_response_for="functionCall",
            released_into=released,
        )

    assert _mentions_tool_call(released)


@pytest.mark.asyncio
async def test_a_gemini_text_stream_is_released_the_same_with_and_without_the_flag() -> None:
    _, at_scan_held, received_held = await _relay("gemini", with_tool_call=False, hold_tool_calls=True, blocks=False)
    _, at_scan_live, received_live = await _relay("gemini", with_tool_call=False, hold_tool_calls=False, blocks=False)

    assert at_scan_held == at_scan_live == received_live
    assert received_held == received_live


@pytest.mark.parametrize("read_size", [1, 7, 61, 200])
@pytest.mark.asyncio
async def test_a_blocked_gemini_function_call_is_held_however_its_stream_is_cut(read_size: int) -> None:
    """The proxy hands this route whole events. A read that stops inside one cannot be read for a function
    call, so it is held too."""
    whole: Final = b"".join(chunk for chunk in _gemini_items(with_tool_call=True) if isinstance(chunk, bytes))
    released: Final[list[object]] = []

    with pytest.raises(ModifyResponseException):
        await _relay(
            "gemini",
            with_tool_call=True,
            hold_tool_calls=True,
            blocks=True,
            wire_items=_cut(whole, read_size),
            blocks_only_tool_calls=True,
            reads_response_for="functionCall",
            released_into=released,
        )

    assert not _mentions_tool_call(released)


@pytest.mark.parametrize("read_size", [1, 7, 61, 200])
@pytest.mark.asyncio
async def test_a_gemini_function_call_event_cut_after_live_text_never_reaches_the_client(read_size: int) -> None:
    *text, call = (chunk for chunk in _gemini_items(with_tool_call=True) if isinstance(chunk, bytes))
    released: Final[list[object]] = []

    with pytest.raises(ModifyResponseException):
        await _relay(
            "gemini",
            with_tool_call=True,
            hold_tool_calls=True,
            blocks=True,
            wire_items=(*text, *_cut(call, read_size)),
            blocks_only_tool_calls=True,
            reads_response_for="functionCall",
            released_into=released,
        )

    assert released == text
    assert not _mentions_tool_call(released)


@pytest.mark.parametrize("read_size", [1, 7, 61, 200])
@pytest.mark.asyncio
async def test_an_allowed_gemini_stream_arrives_byte_for_byte_however_it_is_cut(read_size: int) -> None:
    whole: Final = b"".join(chunk for chunk in _gemini_items(with_tool_call=True) if isinstance(chunk, bytes))

    _, _, received = await _relay(
        "gemini", with_tool_call=True, hold_tool_calls=True, blocks=False, wire_items=_cut(whole, read_size)
    )

    assert b"".join(chunk for chunk in received if isinstance(chunk, bytes)) == whole


class _SdkPart:
    def __init__(self, text: str | None = None, function_call: object | None = None) -> None:
        self.text: Final = text
        self.function_call: Final = function_call


class _SdkContent:
    def __init__(self, parts: object) -> None:
        self.parts: Final = parts


class _SdkCandidate:
    def __init__(self, content: object) -> None:
        self.content: Final = content


class _SdkResponse:
    def __init__(self, candidates: object) -> None:
        self.candidates: Final = candidates


def _gemini_event(chunk: object) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


@pytest.mark.parametrize(
    ("item", "carries_a_function_call"),
    [
        pytest.param(_gemini_event(_gemini_chunk({"text": "hello"})), False, id="sse text"),
        pytest.param(_gemini_event(_gemini_chunk(GEMINI_FUNCTION_CALL)), True, id="sse function call"),
        pytest.param(
            _gemini_event(_gemini_chunk({"text": "hello"})) + _gemini_event(_gemini_chunk(GEMINI_FUNCTION_CALL)),
            True,
            id="sse function call after text in one item",
        ),
        pytest.param(_gemini_event(_gemini_chunk(GEMINI_FUNCTION_CALL)).decode(), True, id="sse function call as str"),
        pytest.param(
            _gemini_event({"candidates": [{"content": {"parts": [{"text": "a"}, GEMINI_FUNCTION_CALL]}}]}),
            True,
            id="sse function call after a text part",
        ),
        pytest.param(
            _gemini_event({"candidates": [_gemini_candidate({"text": "a"}), _gemini_candidate(GEMINI_FUNCTION_CALL)]}),
            True,
            id="sse function call in a second candidate",
        ),
        pytest.param(_gemini_event({"usageMetadata": {"totalTokenCount": 9}}), False, id="sse usage only"),
        pytest.param(
            _gemini_event({"candidates": [{"finishReason": "STOP", "index": 0}]}), False, id="sse finish only"
        ),
        pytest.param(_gemini_event(_gemini_chunk({"executableCode": {"code": "1"}})), False, id="sse server-run code"),
        pytest.param(b": keepalive\n\n", False, id="sse comment"),
        pytest.param(b"event: message\ndata: [DONE]\n\n", False, id="sse done marker"),
        pytest.param(b"", False, id="empty read"),
        pytest.param(b'data: {"candidates": [{"content": {"parts": [{"functi', True, id="sse cut inside an event"),
        pytest.param(b'onCall": {"name": "run_shell"}}]}}]}\n\n', True, id="sse rest of a cut event"),
        pytest.param(b"data: [1, 2]\n\n", True, id="sse payload that is not an object"),
        pytest.param(_gemini_event({"candidates": {"content": {}}}), True, id="sse candidates not a list"),
        pytest.param(
            _gemini_event({"candidates": [{"content": {"parts": {"text": "a"}}}]}), True, id="sse parts not a list"
        ),
        pytest.param(_gemini_chunk({"text": "hello"}), False, id="dict text"),
        pytest.param(_gemini_chunk(GEMINI_FUNCTION_CALL), True, id="dict function call"),
        pytest.param({"usageMetadata": {"totalTokenCount": 9}}, False, id="dict usage only"),
        pytest.param(_SdkResponse([_SdkCandidate(_SdkContent([_SdkPart(text="hello")]))]), False, id="sdk text"),
        pytest.param(
            _SdkResponse([_SdkCandidate(_SdkContent([_SdkPart(function_call={"name": TOOL_NAME})]))]),
            True,
            id="sdk function call",
        ),
        pytest.param(_SdkResponse(None), False, id="sdk usage only"),
        pytest.param(object(), True, id="an object that is not a response"),
        pytest.param(MappingProxyType(_gemini_chunk(GEMINI_FUNCTION_CALL)), True, id="a mapping that is not a dict"),
    ],
)
def test_the_gemini_translation_tells_a_function_call_apart(item: object, carries_a_function_call: bool) -> None:
    assert GoogleGenAIGenerateContentHandler().starts_client_tool_call(item) is carries_a_function_call


@EARLY_RELEASE_MODES
@pytest.mark.asyncio
async def test_a_route_that_cannot_tell_a_tool_call_apart_holds_its_whole_stream_for_the_verdict(mode: str) -> None:
    """The A2A translation scans its stream but has no tool-call detection, so with the flag on nothing of the
    stream is released before the end-of-stream verdict."""
    items, at_scan, received = await _relay(
        "a2a", with_tool_call=True, hold_tool_calls=True, blocks=False, mode=mode, blocks_only_tool_calls=False
    )

    assert at_scan == ()
    assert received == items


@pytest.mark.asyncio
async def test_without_the_flag_a_route_that_cannot_tell_a_tool_call_apart_streams_as_before() -> None:
    items, at_scan, received = await _relay("a2a", with_tool_call=True, hold_tool_calls=False, blocks=False)

    assert at_scan == items
    assert received == items


def test_only_a_translation_that_scans_its_stream_holds_it_whole() -> None:
    """A translation that never scans a stream gives no verdict to hold for, so the hold leaves its stream live."""
    assert A2AGuardrailHandler().starts_client_tool_call({}) is True
    assert OpenAIImageGenerationHandler().starts_client_tool_call(b"partial image") is False
