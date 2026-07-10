"""Regression tests for litellm/responses/streaming_iterator.py.

Two concerns live here:

TTFT stamping (LIT-4185): /v1/responses streaming must stamp
completion_start_time on the first chunk so downstream TTFT consumers
(Prometheus, OTEL, SpendLogs completionStartTime) do not fall back to
completion_start_time = end_time.

Lifecycle-event synthesis (issue #20975): native /responses providers whose
upstream truncates the streaming lifecycle (emitting only
response.output_text.delta ... response.completed) left strict clients like
OpenAI Codex CLI without an "active item" ("OutputTextDelta without active
item"). The live iterators must synthesize the missing setup (response.created,
response.in_progress, response.output_item.added, response.content_part.added)
and teardown (output_text.done, content_part.done, output_item.done) events,
pass an already-complete sequence through unchanged (idempotency), and run the
post-call streaming deployment hook BEFORE the gap filler accumulates deltas so
a hook that redacts delta text is not bypassed on the synthesized *.done events.

The #20975 tests drive the REAL ResponsesAPIStreamingIterator /
SyncResponsesAPIStreamingIterator with the REAL OpenAIResponsesAPIConfig,
feeding a dependency-injected fake SSE byte stream (no monkeypatching of the
code under test).
"""

import json
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
    _obj_get,
    _ResponsesLifecycleGapFiller,
    _safe_int,
)
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)

EV = ResponsesAPIStreamEvents
E = EV  # shorthand


# ---------------------------------------------------------------------------
# TTFT stamping (LIT-4185)
# ---------------------------------------------------------------------------


def _sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _make_iterator(
    *,
    sse_events: list[bytes],
    logging_obj: LiteLLMLoggingObj,
) -> ResponsesAPIStreamingIterator:
    async def aiter_bytes():
        for evt in sse_events:
            yield evt

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = aiter_bytes

    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
    mock_responses_api_response.id = "resp_ttft"

    def _transform(model, parsed_chunk, logging_obj):
        evt_type = parsed_chunk.get("type")
        if evt_type == "response.completed":
            completed = Mock(spec=ResponseCompletedEvent)
            completed.type = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
            completed.response = mock_responses_api_response
            return completed
        stub = Mock()
        stub.type = evt_type
        return stub

    mock_config.transform_streaming_response.side_effect = _transform

    return ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-4o-mini",
        responses_api_provider_config=mock_config,
        logging_obj=logging_obj,
        litellm_metadata={},
        custom_llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_responses_streaming_stamps_completion_start_time_on_first_chunk():
    """Without the fix, `logging_obj.completion_start_time` stays None across the
    entire stream and _success_handler_helper_fn falls back to end_time — collapsing
    the reported TTFT to full generation time."""
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.completion_start_time = None
    logging_obj.model_call_details = {"litellm_params": {}}
    stamped: list[datetime] = []

    def _update(*, completion_start_time):
        stamped.append(completion_start_time)
        logging_obj.completion_start_time = completion_start_time
        logging_obj.model_call_details["completion_start_time"] = completion_start_time

    logging_obj._update_completion_start_time.side_effect = _update

    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.created"}),
            _sse_event({"type": "response.output_text.delta", "delta": "hi"}),
            _sse_event({"type": "response.completed"}),
        ],
        logging_obj=logging_obj,
    )

    async for _ in iterator:
        pass

    assert len(stamped) == 1, (
        f"Expected exactly one first-chunk stamp; got {len(stamped)}. "
        "Later chunks must not re-stamp completion_start_time."
    )
    assert isinstance(stamped[0], datetime)


@pytest.mark.asyncio
async def test_responses_streaming_does_not_reset_prior_completion_start_time():
    """If `completion_start_time` is already set (e.g. by an outer wrapper), the
    iterator must not overwrite it — otherwise TTFT would collapse to
    time-to-last-chunk under contention."""
    prior = datetime(2020, 1, 1, 0, 0, 0)
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.completion_start_time = prior
    logging_obj.model_call_details = {"litellm_params": {}}

    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.created"}),
            _sse_event({"type": "response.completed"}),
        ],
        logging_obj=logging_obj,
    )

    async for _ in iterator:
        pass

    logging_obj._update_completion_start_time.assert_not_called()
    assert logging_obj.completion_start_time == prior


# ---------------------------------------------------------------------------
# Lifecycle-event synthesis (issue #20975)
# ---------------------------------------------------------------------------


def _response_body(status: str) -> Dict[str, Any]:
    return {
        "id": "resp_real_upstream",
        "object": "response",
        "created_at": 1_700_000_000,
        "status": status,
        "model": "gpt-5",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello world", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _sse_frames(events: List[Dict[str, Any]]) -> List[bytes]:
    """One `data: {...}\\n\\n` SSE frame per event, plus a terminating [DONE]."""
    frames = [f"data: {json.dumps(evt)}\n\n".encode("utf-8") for evt in events]
    frames.append(b"data: [DONE]\n\n")
    return frames


class _FakeStreamResponse:
    """Minimal stand-in for httpx.Response exposing (a)iter_bytes over fixed frames."""

    def __init__(self, frames: List[bytes]):
        self.headers: Dict[str, str] = {}
        self._frames = frames

    async def aiter_bytes(self):
        for frame in self._frames:
            yield frame

    def iter_bytes(self):
        for frame in self._frames:
            yield frame


def _make_logging_obj() -> Any:
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.completion_start_time = None
    return logging_obj


def _iterator(events: List[Dict[str, Any]], *, sync: bool) -> Any:
    response = _FakeStreamResponse(_sse_frames(events))
    cls = SyncResponsesAPIStreamingIterator if sync else ResponsesAPIStreamingIterator
    return cls(
        response=response,
        model="gpt-5",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=_make_logging_obj(),
        litellm_metadata={"model_info": {"id": "model_123"}},
        custom_llm_provider="openai",
    )


async def _drive(events: List[Dict[str, Any]], *, sync: bool) -> List[Any]:
    with (
        patch("asyncio.create_task"),
        patch("litellm.responses.streaming_iterator.executor"),
    ):
        iterator = _iterator(events, sync=sync)
        collected: List[Any] = []
        if sync:
            for chunk in iterator:
                collected.append(chunk)
        else:
            async for chunk in iterator:
                collected.append(chunk)
    return collected


def _types(events: List[Any]) -> List[Any]:
    return [getattr(e, "type", None) for e in events]


# ----- truncated upstream (the copilot / ollama / Azure case) -----

_TRUNCATED_TEXT_EVENTS: List[Dict[str, Any]] = [
    {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "Hello",
    },
    {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": " world",
    },
    {"type": "response.completed", "response": _response_body("completed")},
]

_FULL_TEXT_EVENTS: List[Dict[str, Any]] = [
    {"type": "response.created", "response": _response_body("in_progress")},
    {"type": "response.in_progress", "response": _response_body("in_progress")},
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "id": "msg_1",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        },
    },
    {
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    },
    {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "Hello world",
    },
    {
        "type": "response.output_text.done",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "text": "Hello world",
    },
    {
        "type": "response.content_part.done",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "Hello world", "annotations": []},
    },
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello world", "annotations": []}],
        },
    },
    {"type": "response.completed", "response": _response_body("completed")},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_truncated_text_stream_synthesizes_full_lifecycle(sync):
    collected = await _drive(_TRUNCATED_TEXT_EVENTS, sync=sync)
    types = _types(collected)

    assert types == [
        E.RESPONSE_CREATED,
        E.RESPONSE_IN_PROGRESS,
        E.OUTPUT_ITEM_ADDED,
        E.CONTENT_PART_ADDED,
        E.OUTPUT_TEXT_DELTA,
        E.OUTPUT_TEXT_DELTA,
        E.OUTPUT_TEXT_DONE,
        E.CONTENT_PART_DONE,
        E.OUTPUT_ITEM_DONE,
        E.RESPONSE_COMPLETED,
    ], types

    # openers must anchor to the same item_id / indices as the deltas
    output_item_added = collected[2]
    content_part_added = collected[3]
    assert output_item_added.item.id == "msg_1"
    assert content_part_added.item_id == "msg_1"
    assert content_part_added.output_index == 0
    assert content_part_added.content_index == 0

    # teardown text must equal the concatenation of streamed deltas
    output_text_done = collected[6]
    assert output_text_done.text == "Hello world"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_complete_stream_passes_through_without_duplication(sync):
    collected = await _drive(_FULL_TEXT_EVENTS, sync=sync)
    types = _types(collected)

    # byte-for-byte: same event types, same count, nothing injected
    assert types == [evt["type"] for evt in _FULL_TEXT_EVENTS], types
    assert len(collected) == len(_FULL_TEXT_EVENTS)
    # no duplicated openers
    assert types.count(E.RESPONSE_CREATED) == 1
    assert types.count(E.OUTPUT_ITEM_ADDED) == 1
    assert types.count(E.CONTENT_PART_ADDED) == 1
    assert types.count(E.OUTPUT_ITEM_DONE) == 1


@pytest.mark.asyncio
async def test_truncated_function_call_stream_synthesizes_item_lifecycle():
    events = [
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": '{"city":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": '"NYC"}',
        },
        {"type": "response.completed", "response": _response_body("completed")},
    ]
    collected = await _drive(events, sync=False)
    types = _types(collected)

    assert types == [
        E.RESPONSE_CREATED,
        E.RESPONSE_IN_PROGRESS,
        E.OUTPUT_ITEM_ADDED,
        E.FUNCTION_CALL_ARGUMENTS_DELTA,
        E.FUNCTION_CALL_ARGUMENTS_DELTA,
        E.FUNCTION_CALL_ARGUMENTS_DONE,
        E.OUTPUT_ITEM_DONE,
        E.RESPONSE_COMPLETED,
    ], types

    # function_call items have NO content part
    assert E.CONTENT_PART_ADDED not in types
    assert E.CONTENT_PART_DONE not in types

    output_item_added = collected[2]
    assert output_item_added.item.type == "function_call"
    assert output_item_added.item.id == "fc_1"

    args_done = collected[5]
    assert args_done.arguments == '{"city":"NYC"}'


@pytest.mark.asyncio
async def test_synthesized_events_survive_proxy_serialization():
    """
    The proxy serializes each event with model_dump_json(exclude_none=True,
    exclude_unset=True). Synthesized events must set their required fields
    explicitly so nothing load-bearing is stripped off the wire.
    """
    collected = await _drive(_TRUNCATED_TEXT_EVENTS, sync=False)

    required_by_type = {
        E.OUTPUT_ITEM_ADDED: ["type", "output_index", "item"],
        E.CONTENT_PART_ADDED: [
            "type",
            "item_id",
            "output_index",
            "content_index",
            "part",
        ],
        E.OUTPUT_TEXT_DONE: [
            "type",
            "item_id",
            "output_index",
            "content_index",
            "text",
        ],
        E.CONTENT_PART_DONE: [
            "type",
            "item_id",
            "output_index",
            "content_index",
            "part",
        ],
        E.OUTPUT_ITEM_DONE: ["type", "output_index", "item"],
    }

    seen_types = set()
    for event in collected:
        etype = getattr(event, "type", None)
        if etype not in required_by_type:
            continue
        seen_types.add(etype)
        wire = json.loads(event.model_dump_json(exclude_none=True, exclude_unset=True))
        for field in required_by_type[etype]:
            assert field in wire, f"{etype} lost required field {field}: {wire}"

    # all synthesized wrapper events were exercised
    assert seen_types == set(required_by_type.keys())


class _RedactingDeploymentHook:
    """A streaming deployment hook that redacts output_text delta content."""

    REDACTION = "[REDACTED]"

    async def async_post_call_streaming_deployment_hook(self, *, request_data, response_chunk, call_type):
        if getattr(response_chunk, "type", None) == E.OUTPUT_TEXT_DELTA:
            response_chunk.delta = self.REDACTION
        return response_chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_streaming_hook_governs_synthesized_teardown(sync):
    """
    A post-call streaming deployment hook that redacts response.output_text.delta
    must also govern the SYNTHESIZED teardown. The gap filler accumulates the
    post-hook delta, so output_text.done / content_part.done / output_item.done
    carry the redacted text, never the raw provider text (issue #20975 review:
    the pre-hook accumulation leaked redacted content through the done events).
    """
    redacted = _RedactingDeploymentHook.REDACTION * 2  # two deltas
    with patch.object(litellm, "callbacks", [_RedactingDeploymentHook()]):
        collected = await _drive(_TRUNCATED_TEXT_EVENTS, sync=sync)

    by_type: Dict[Any, List[Any]] = {}
    for event in collected:
        by_type.setdefault(getattr(event, "type", None), []).append(event)

    # client-visible deltas are redacted
    assert [d.delta for d in by_type[E.OUTPUT_TEXT_DELTA]] == [
        _RedactingDeploymentHook.REDACTION,
        _RedactingDeploymentHook.REDACTION,
    ]

    # synthesized teardown reflects the post-hook (redacted) accumulation
    assert by_type[E.OUTPUT_TEXT_DONE][0].text == redacted
    assert by_type[E.CONTENT_PART_DONE][0].part.text == redacted
    assert by_type[E.OUTPUT_ITEM_DONE][0].item.content[0].text == redacted

    # the raw provider text never leaks anywhere in the stream
    assert all(getattr(e, "text", None) != "Hello world" for e in collected)


# ----- truncated refusal stream -----

_TRUNCATED_REFUSAL_EVENTS: List[Dict[str, Any]] = [
    {
        "type": "response.refusal.delta",
        "item_id": "msg_r",
        "output_index": 0,
        "content_index": 0,
        "delta": "I can",
    },
    {
        "type": "response.refusal.delta",
        "item_id": "msg_r",
        "output_index": 0,
        "content_index": 0,
        "delta": "not help",
    },
    {"type": "response.completed", "response": _response_body("completed")},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_truncated_refusal_stream_synthesizes_lifecycle(sync):
    collected = await _drive(_TRUNCATED_REFUSAL_EVENTS, sync=sync)
    types = _types(collected)

    assert types == [
        E.RESPONSE_CREATED,
        E.RESPONSE_IN_PROGRESS,
        E.OUTPUT_ITEM_ADDED,
        E.CONTENT_PART_ADDED,
        E.REFUSAL_DELTA,
        E.REFUSAL_DELTA,
        E.REFUSAL_DONE,
        E.CONTENT_PART_DONE,
        E.OUTPUT_ITEM_DONE,
        E.RESPONSE_COMPLETED,
    ], types

    # the synthesized content part is a refusal part, not output_text
    assert collected[3].part.type == "refusal"
    # teardown carries the accumulated refusal text at every level
    assert collected[6].refusal == "I cannot help"
    assert collected[7].part.refusal == "I cannot help"
    assert collected[8].item.content[0].refusal == "I cannot help"


def test_obj_get_handles_dict_object_and_none():
    assert _obj_get({"a": 1}, "a") == 1
    assert _obj_get({"a": 1}, "missing", "d") == "d"
    assert _obj_get(None, "a", "d") == "d"

    class _Obj:
        x = 5

    assert _obj_get(_Obj(), "x") == 5
    assert _obj_get(_Obj(), "y", "fallback") == "fallback"


def test_safe_int_narrows_dynamic_values():
    assert _safe_int(3, 0) == 3
    assert _safe_int(True, 9) == 9  # bool is not an accepted int
    assert _safe_int("5", 0) == 5
    assert _safe_int("nope", 7) == 7
    assert _safe_int(1.5, 4) == 4


def test_gap_filler_passes_unknown_event_through():
    gap_filler = _ResponsesLifecycleGapFiller(model="m", response_id="resp_x")
    event = {"type": "response.some_unhandled_event"}
    assert gap_filler.expand(event) == (event,)
