"""
Full-coverage unit tests for AnthropicResponsesStreamWrapper.

Complements `test_streaming_iterator_reasoning.py` (which covers the
signature_delta emission for the encrypted-reasoning round-trip) by exercising
the rest of the wrapper's event handling:

* function_call output_item.added/done and arguments-delta translation
* response.completed branches: status=completed/incomplete, usage extraction,
  cache token deltas, and the tool_use stop_reason override
* Defensive paths: event with no type, output_item.added with item=None
* Iterator-level paths: upstream exception draining and the SSE byte wrapper
"""

import asyncio
import json

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


class _AsyncEventStream:
    """Minimal async iterator over a static list of event dicts."""

    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class _RaisingEventStream:
    """Async iterator that yields one event then raises — exercises the
    wrapper's except handler in __anext__."""

    def __init__(self, events, exc):
        self._events = list(events)
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._events:
            return self._events.pop(0)
        raise self._exc


def _collect(wrapper):
    async def _drive():
        out = []
        async for chunk in wrapper:
            out.append(chunk)
        return out

    return asyncio.run(_drive())


# ---------------------------------------------------------------------------
# function_call branches
# ---------------------------------------------------------------------------


class _Resp:
    """Attribute-style response object used by response.completed handling.
    The wrapper uses getattr() (not dict access) on the embedded response."""

    def __init__(self, status="completed", output=None, usage=None):
        self.status = status
        self.output = output or []
        self.usage = usage


class _OutputItem:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


def _function_call_events():
    call_id = "call_42"
    item_id = "fc_xyz"
    return [
        {"type": "response.created"},
        {
            "type": "response.output_item.added",
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": "get_weather",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "delta": '{"city":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "delta": '"SF"}',
        },
        {
            "type": "response.output_item.done",
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": "get_weather",
                "arguments": '{"city":"SF"}',
            },
        },
        {
            "type": "response.completed",
            "response": _Resp(
                status="completed",
                output=[
                    _OutputItem(type="function_call", id=item_id, name="get_weather")
                ],
            ),
        },
    ]


def test_function_call_emits_tool_use_start_and_input_json_deltas():
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_AsyncEventStream(_function_call_events()), model="gpt-5.5"
    )
    chunks = _collect(wrapper)

    starts = [
        c
        for c in chunks
        if c.get("type") == "content_block_start"
        and c["content_block"]["type"] == "tool_use"
    ]
    assert len(starts) == 1
    assert starts[0]["content_block"]["id"] == "call_42"
    assert starts[0]["content_block"]["name"] == "get_weather"

    json_deltas = [
        c["delta"]["partial_json"]
        for c in chunks
        if c.get("type") == "content_block_delta"
        and c["delta"].get("type") == "input_json_delta"
    ]
    assert "".join(json_deltas) == '{"city":"SF"}'


def test_function_call_in_output_overrides_stop_reason_to_tool_use():
    """A response.completed whose output array contains a function_call
    item must flip stop_reason from end_turn → tool_use."""
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_AsyncEventStream(_function_call_events()), model="gpt-5.5"
    )
    chunks = _collect(wrapper)

    deltas = [c for c in chunks if c.get("type") == "message_delta"]
    assert deltas, "expected a message_delta"
    assert deltas[-1]["delta"]["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# response.completed: usage + incomplete + cache tokens
# ---------------------------------------------------------------------------


def test_completed_incomplete_status_maps_to_max_tokens():
    events = [
        {"type": "response.created"},
        {
            "type": "response.completed",
            "response": _Resp(status="incomplete"),
        },
    ]
    chunks = _collect(
        AnthropicResponsesStreamWrapper(
            responses_stream=_AsyncEventStream(events), model="gpt-5.5"
        )
    )
    delta = next(c for c in chunks if c.get("type") == "message_delta")
    assert delta["delta"]["stop_reason"] == "max_tokens"


def test_completed_propagates_usage_and_cache_tokens():
    class _Usage:
        input_tokens = 123
        output_tokens = 45
        cache_creation_input_tokens = 9
        cache_read_input_tokens = 6

    class _Resp:
        status = "completed"
        usage = _Usage()
        output = []

    events = [
        {"type": "response.created"},
        {"type": "response.completed", "response": _Resp()},
    ]
    chunks = _collect(
        AnthropicResponsesStreamWrapper(
            responses_stream=_AsyncEventStream(events), model="gpt-5.5"
        )
    )
    delta = next(c for c in chunks if c.get("type") == "message_delta")
    usage = delta["usage"]
    assert usage["input_tokens"] == 123
    assert usage["output_tokens"] == 45
    assert usage["cache_creation_input_tokens"] == 9
    assert usage["cache_read_input_tokens"] == 6


def test_completed_without_cache_tokens_omits_cache_keys():
    class _Usage:
        input_tokens = 5
        output_tokens = 3

    class _Resp:
        status = "completed"
        usage = _Usage()
        output = []

    events = [
        {"type": "response.created"},
        {"type": "response.completed", "response": _Resp()},
    ]
    chunks = _collect(
        AnthropicResponsesStreamWrapper(
            responses_stream=_AsyncEventStream(events), model="gpt-5.5"
        )
    )
    delta = next(c for c in chunks if c.get("type") == "message_delta")
    assert "cache_creation_input_tokens" not in delta["usage"]
    assert "cache_read_input_tokens" not in delta["usage"]


# ---------------------------------------------------------------------------
# Defensive / edge cases
# ---------------------------------------------------------------------------


def test_event_without_type_is_ignored():
    """An event dict missing 'type' must early-return; no chunks produced
    beyond the fallback message_start."""
    events = [
        {"not_type": "garbage"},
        {
            "type": "response.completed",
            "response": {"status": "completed", "output": [], "usage": None},
        },
    ]
    chunks = _collect(
        AnthropicResponsesStreamWrapper(
            responses_stream=_AsyncEventStream(events), model="gpt-5.5"
        )
    )
    # We expect message_start, then message_delta + message_stop. The garbage
    # event is silently dropped — no extra content_block_* events.
    types = [c.get("type") for c in chunks]
    assert "message_start" in types
    assert types.count("content_block_start") == 0


def test_output_item_added_with_no_item_payload_is_ignored():
    events = [
        {"type": "response.created"},
        {"type": "response.output_item.added"},  # no `item` key
        {
            "type": "response.completed",
            "response": {"status": "completed", "output": [], "usage": None},
        },
    ]
    chunks = _collect(
        AnthropicResponsesStreamWrapper(
            responses_stream=_AsyncEventStream(events), model="gpt-5.5"
        )
    )
    assert not any(c.get("type") == "content_block_start" for c in chunks)


# ---------------------------------------------------------------------------
# Iterator-level: exception handling + SSE byte wrapper
# ---------------------------------------------------------------------------


def test_upstream_exception_is_swallowed_and_remaining_chunks_drained():
    """When the upstream Responses stream raises mid-iteration, the wrapper
    logs the error, drains the queue, and finishes cleanly with
    StopAsyncIteration — it must not propagate the exception to the caller."""
    events = [
        {"type": "response.created"},
        {
            "type": "response.output_item.added",
            "item": {"id": "msg_1", "type": "message"},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": "partial",
        },
    ]
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_RaisingEventStream(events, RuntimeError("upstream blew up")),
        model="gpt-5.5",
    )
    chunks = _collect(wrapper)
    # We at least got message_start and the partial text_delta before the blow-up;
    # no exception escaped.
    types = [c.get("type") for c in chunks]
    assert "message_start" in types
    assert any(
        c.get("type") == "content_block_delta"
        and c["delta"].get("type") == "text_delta"
        for c in chunks
    )


def test_async_anthropic_sse_wrapper_yields_event_data_bytes():
    """The SSE byte wrapper must format each chunk as `event: <type>\\n
    data: <json>\\n\\n` and encode to bytes."""
    events = [
        {"type": "response.created"},
        {
            "type": "response.completed",
            "response": {"status": "completed", "output": [], "usage": None},
        },
    ]
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_AsyncEventStream(events), model="gpt-5.5"
    )

    async def _drive():
        out = []
        async for b in wrapper.async_anthropic_sse_wrapper():
            out.append(b)
        return out

    payloads = asyncio.run(_drive())
    assert payloads, "expected SSE byte chunks"
    # Every chunk is bytes shaped like b"event: <type>\ndata: {json}\n\n"
    for b in payloads:
        assert isinstance(b, bytes)
        text = b.decode()
        assert text.startswith("event: ")
        assert "\ndata: " in text
        assert text.endswith("\n\n")
        # JSON body parses
        data_line = text.split("\ndata: ", 1)[1].rstrip("\n")
        json.loads(data_line)
