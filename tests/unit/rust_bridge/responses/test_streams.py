import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator, SyncResponsesAPIStreamingIterator
from litellm.rust_bridge.execution import Complete, Step, StreamChunk, StreamOpen, drive, run
from litellm.rust_bridge.responses.route_host import stream_head
from litellm.rust_bridge.responses.streams import StreamSource, open_async, open_sync
from litellm.types.llms.openai import ResponsesAPIStreamEvents

_HEADERS: Final = (("content-type", "text/event-stream"), ("x-request-id", "req_native"))
_RESPONSE: Final = {
    "id": "resp_native",
    "object": "response",
    "created_at": 1,
    "model": "gpt-4o",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_native",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Paris", "annotations": []}],
        }
    ],
}
_EVENTS: Final = (
    {
        "type": "response.output_text.delta",
        "sequence_number": 0,
        "item_id": "msg_native",
        "output_index": 0,
        "content_index": 0,
        "delta": "Paris",
    },
    {"type": "response.completed", "sequence_number": 1, "response": _RESPONSE},
)


def _sse(event: Mapping[str, object]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


@dataclass(slots=True)
class _Execution:
    """Opens with `head`, then yields each chunk before completing"""

    head: object
    chunks: tuple[object, ...]
    sent: list[object] = field(default_factory=list)  # mutable-ok: records what the driver fed back
    closed: bool = False

    def send(self, value: object) -> Step:
        self.sent.append(value)
        index: Final = len(self.sent) - 2
        if index < 0:
            return StreamOpen(self.head)
        return StreamChunk(self.chunks[index]) if index < len(self.chunks) else Complete(None)

    def throw(self, error: BaseException) -> Step:
        raise error

    def close(self) -> None:
        self.closed = True


def _logger() -> Logging:
    logger: Final = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="responses",
        start_time=datetime.now(),  # noqa: DTZ005  # Logging start times are naive local datetimes
        litellm_call_id="native-stream",
        function_id="native-stream",
    )
    logger.update_environment_variables(litellm_params={}, optional_params={}, model="gpt-4o")
    return logger


def _request(logger: object, **fields: object) -> dict[str, object]:
    return {"model": "openai/gpt-4o", "input": "hi", "stream": True, "litellm_logging_obj": logger, **fields}


def _execution(logger: Logging, chunks: tuple[object, ...] = tuple(map(_sse, _EVENTS))) -> _Execution:
    return _Execution(head=stream_head(_HEADERS, _request(logger)), chunks=chunks)


@pytest.mark.asyncio
async def test_drive_streams_native_bytes_as_typed_events() -> None:
    logger: Final = _logger()
    execution: Final = _execution(logger)

    stream: Final = await drive(execution, open_async)

    assert isinstance(stream, ResponsesAPIStreamingIterator)
    events: Final = [event async for event in stream]
    assert [event.type for event in events] == [
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    ]
    assert events[0].delta == "Paris"
    assert events[-1].response.output[0].content[0].text == "Paris"
    assert execution.closed
    assert logger.model_call_details["litellm_params"]["aresponses"] is True
    hidden: Final = stream._hidden_params
    assert hidden["additional_headers"]["llm_provider-x-request-id"] == "req_native"
    assert (hidden["litellm_call_id"], hidden["litellm_model_name"]) == ("native-stream", "openai/gpt-4o")


def test_sync_stream_streams_native_bytes_as_typed_events() -> None:
    logger: Final = _logger()
    execution: Final = _execution(logger)

    stream: Final = run(execution, open_sync)

    assert isinstance(stream, SyncResponsesAPIStreamingIterator)
    assert [event.type for event in stream] == [
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    ]
    assert execution.closed
    assert logger.model_call_details["litellm_params"]["aresponses"] is False


@pytest.mark.asyncio
async def test_a_head_without_a_stream_source_closes_the_execution() -> None:
    execution: Final = _Execution(head={"additional_headers": {}}, chunks=())

    with pytest.raises(TypeError, match="no stream source"):
        await drive(execution, open_async)

    assert execution.closed


@pytest.mark.asyncio
async def test_a_chunk_that_is_not_bytes_is_rejected() -> None:
    stream: Final = await drive(_execution(_logger(), chunks=("event: text\n\n",)), open_async)

    assert isinstance(stream, ResponsesAPIStreamingIterator)
    with pytest.raises(TypeError, match="chunk is bytes, got str"):
        _ = [event async for event in stream]


def test_stream_head_projects_headers_and_builds_the_iterator_request() -> None:
    metadata: Final = {"model_info": {"id": "deployment"}}
    request: Final = _request(_logger(), litellm_metadata=metadata, temperature=None, max_output_tokens=12)

    head: Final = stream_head(_HEADERS, request)

    assert head["additional_headers"]["llm_provider-x-request-id"] == "req_native"
    source: Final = head["source"]
    assert isinstance(source, StreamSource)
    assert (source.model, source.custom_llm_provider) == ("gpt-4o", "openai")
    assert source.litellm_metadata is metadata
    assert source.request_data["max_output_tokens"] == 12
    assert "temperature" not in source.request_data
    assert "litellm_logging_obj" not in source.request_data["litellm_params"]


def test_stream_head_without_a_call_logger_raises() -> None:
    with pytest.raises(RuntimeError, match="call logger"):
        stream_head(_HEADERS, _request(None))
