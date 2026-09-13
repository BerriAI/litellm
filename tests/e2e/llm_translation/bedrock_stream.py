from typing import Final

from e2e_http import StreamingResponse, require_successful_call
from pydantic import BaseModel


class _TextDelta(BaseModel):
    text: str = ""
    stop_reason: str | None = None


class _ContentBlockDelta(BaseModel):
    delta: _TextDelta


class _MessageStart(BaseModel):
    role: str


class _MessageStop(BaseModel):
    stopReason: str


class _ConverseUsage(BaseModel):
    inputTokens: int
    outputTokens: int
    totalTokens: int


class _Metadata(BaseModel):
    usage: _ConverseUsage


class _ConverseEvent(BaseModel):
    messageStart: _MessageStart | None = None
    contentBlockDelta: _ContentBlockDelta | None = None
    messageStop: _MessageStop | None = None
    metadata: _Metadata | None = None


class _InvokeUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class _InvokeMessage(BaseModel):
    role: str
    usage: _InvokeUsage


class _InvokeEvent(BaseModel):
    type: str
    message: _InvokeMessage | None = None
    delta: _TextDelta | None = None
    usage: _InvokeUsage | None = None


def _require_native_stream(result: StreamingResponse) -> None:
    require_successful_call(result, expected_provider="bedrock")
    assert "application/vnd.amazon.eventstream" in (result.content_type or "").lower(), result.content_type
    assert result.stream_error is None, result.stream_error
    assert result.stream_events, "Bedrock stream returned no decoded events"


def assert_converse_stream(result: StreamingResponse) -> None:
    _require_native_stream(result)
    events: Final = tuple(_ConverseEvent.model_validate_json(event) for event in result.stream_events)
    starts: Final = tuple(i for i, event in enumerate(events) if event.messageStart is not None)
    deltas: Final = tuple(i for i, event in enumerate(events) if event.contentBlockDelta is not None)
    stops: Final = tuple(i for i, event in enumerate(events) if event.messageStop is not None)
    metadata: Final = tuple(i for i, event in enumerate(events) if event.metadata is not None)
    assert len(starts) == len(stops) == len(metadata) == 1, "Converse requires messageStart, messageStop and metadata"
    assert deltas, "Converse stream returned no content deltas"
    assert starts[0] < deltas[0] <= deltas[-1] < stops[0] < metadata[0] == len(events) - 1, "Converse event order"
    start: Final = events[starts[0]].messageStart
    stop: Final = events[stops[0]].messageStop
    tail: Final = events[metadata[0]].metadata
    assert start is not None and start.role == "assistant", "Converse stream has no assistant"
    text: Final = "".join(event.contentBlockDelta.delta.text for event in events if event.contentBlockDelta is not None)
    assert text.strip(), "Converse stream returned empty text"
    assert stop is not None and stop.stopReason in {"end_turn", "max_tokens"}, (
        "Converse stream has no valid stop reason"
    )
    assert tail is not None and tail.usage.inputTokens > 0 and tail.usage.outputTokens > 0, "Converse usage is missing"
    assert tail.usage.totalTokens == tail.usage.inputTokens + tail.usage.outputTokens, "Converse usage does not add up"


def assert_invoke_stream(result: StreamingResponse) -> None:
    _require_native_stream(result)
    events: Final = tuple(_InvokeEvent.model_validate_json(event) for event in result.stream_events)
    starts: Final = tuple(i for i, event in enumerate(events) if event.type == "message_start")
    deltas: Final = tuple(i for i, event in enumerate(events) if event.type == "content_block_delta")
    stops: Final = tuple(i for i, event in enumerate(events) if event.type == "message_stop")
    finishes: Final = tuple(
        i
        for i, event in enumerate(events)
        if event.type == "message_delta" and event.delta is not None and event.delta.stop_reason is not None
    )
    assert len(starts) == len(stops) == len(finishes) == 1, "Invoke requires message_start, finish and message_stop"
    assert deltas, "Invoke stream returned no content deltas"
    assert starts[0] < deltas[0] <= deltas[-1] < finishes[0] < stops[0] == len(events) - 1, "Invoke event order"
    message: Final = events[starts[0]].message
    assert message is not None and message.role == "assistant", "Invoke stream has no assistant"
    assert message.usage.input_tokens > 0, "Invoke input usage is missing"
    text: Final = "".join(
        event.delta.text for event in events if event.type == "content_block_delta" and event.delta is not None
    )
    assert text.strip(), "Invoke stream returned empty text"
    finish: Final = events[finishes[0]]
    assert finish.delta is not None and finish.delta.stop_reason in {"end_turn", "max_tokens"}, "Invoke stop reason"
    assert finish.usage is not None and finish.usage.output_tokens > 0, "Invoke output usage is missing"
