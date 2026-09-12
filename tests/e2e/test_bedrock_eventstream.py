import base64
import json
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pytest
from bedrock_eventstream import decode_bedrock_stream
from botocore.eventstream import ChecksumMismatch
from e2e_http import StreamingResponse, streaming_outcome
from llm_translation.bedrock_stream import assert_converse_stream, assert_invoke_stream
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _frame(payload: str, event_type: str, message_type: str = "event") -> bytes:
    headers: Final = {
        ":message-type": message_type,
        ":event-type" if message_type == "event" else ":exception-type": event_type,
        ":content-type": "application/json",
    }
    encoded_headers: Final = b"".join(
        bytes([len(key)]) + key.encode() + b"\x07" + struct.pack("!H", len(value)) + value.encode()
        for key, value in headers.items()
    )
    prelude: Final = struct.pack("!II", 16 + len(encoded_headers) + len(payload.encode()), len(encoded_headers))
    message: Final = prelude + struct.pack("!I", zlib.crc32(prelude)) + encoded_headers + payload.encode()
    return message + struct.pack("!I", zlib.crc32(message))


_CONVERSE: Final = (
    '{"messageStart":{"role":"assistant"}}',
    '{"contentBlockDelta":{"delta":{"text":"Hello"},"contentBlockIndex":0}}',
    '{"contentBlockStop":{"contentBlockIndex":0}}',
    '{"messageStop":{"stopReason":"end_turn"}}',
    '{"metadata":{"usage":{"inputTokens":10,"outputTokens":1,"totalTokens":11}}}',
)
_INVOKE: Final = (
    '{"type":"message_start","message":{"role":"assistant","usage":{"input_tokens":10,"output_tokens":0}}}',
    '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
    '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}',
    '{"type":"message_stop"}',
)


@dataclass(frozen=True, slots=True)
class _BinaryResponse:
    chunks: tuple[bytes, ...]
    status_code: int = 200
    text: str = ""
    headers = MappingProxyType({"content-type": "application/vnd.amazon.eventstream"})

    def iter_content(self, chunk_size: int | None = 1) -> Iterator[bytes]:
        return iter(self.chunks)

    def iter_lines(self) -> Iterator[bytes]:
        raise AssertionError("binary Bedrock streams must not be consumed as text lines")


def _stream(events: tuple[str, ...]) -> StreamingResponse:
    return StreamingResponse(
        status_code=200,
        content_type="application/vnd.amazon.eventstream",
        body="<streamed>",
        stream_events=list(events),
        chunks=len(events),
    )


class TestBedrockEventStream:
    @pytest.mark.parametrize("size", [1, 7, 4096])
    def test_converse_binary_frames_survive_arbitrary_transport_splits(self, size: int) -> None:
        wire: Final = b"".join(
            _frame(json.dumps(value), key)
            for payload in _CONVERSE
            for key, value in _JSON_OBJECT.validate_json(payload).items()
        )
        chunks: Final = tuple(wire[i : i + size] for i in range(0, len(wire), size))
        result: Final = streaming_outcome(_BinaryResponse(chunks), True, sent_at=0.0, clock=lambda: 1.0)
        assert result.chunks == len(_CONVERSE)
        assert_converse_stream(result)

    def test_invoke_payload_is_decoded_from_its_base64_envelope(self) -> None:
        wire: Final = b"".join(
            _frame(json.dumps({"bytes": base64.b64encode(event.encode()).decode()}), "chunk") for event in _INVOKE
        )
        result: Final = streaming_outcome(_BinaryResponse((wire,)), True, sent_at=0.0)
        assert result.stream_events == list(_INVOKE)
        assert_invoke_stream(result)

    def test_binary_provider_exception_is_not_counted_as_a_successful_stream(self) -> None:
        wire: Final = _frame('{"message":"unavailable"}', "serviceUnavailableException", "exception")
        result: Final = streaming_outcome(_BinaryResponse((wire,)), True, sent_at=0.0)
        assert result.stream_error == 'serviceUnavailableException: {"message":"unavailable"}'
        with pytest.raises(AssertionError, match="serviceUnavailableException"):
            assert_converse_stream(result)

    def test_corrupt_binary_frame_is_rejected(self) -> None:
        wire: Final = _frame('{"role":"assistant"}', "messageStart")
        with pytest.raises(ChecksumMismatch):
            tuple(decode_bedrock_stream((wire[:-1] + bytes([wire[-1] ^ 1]),)))


class TestNativeStreamAssertions:
    @pytest.mark.parametrize("missing", [0, 1, 3, 4])
    def test_converse_rejects_missing_content_or_completion(self, missing: int) -> None:
        with pytest.raises(AssertionError):
            assert_converse_stream(_stream(tuple(event for i, event in enumerate(_CONVERSE) if i != missing)))

    @pytest.mark.parametrize("missing", [0, 1, 2, 3])
    def test_invoke_rejects_missing_content_or_completion(self, missing: int) -> None:
        with pytest.raises(AssertionError):
            assert_invoke_stream(_stream(tuple(event for i, event in enumerate(_INVOKE) if i != missing)))

    @pytest.mark.parametrize("events", [_CONVERSE, _INVOKE])
    def test_empty_text_does_not_pass_because_events_arrived(self, events: tuple[str, ...]) -> None:
        assertion: Final = assert_converse_stream if events == _CONVERSE else assert_invoke_stream
        with pytest.raises(AssertionError, match="empty text"):
            assertion(_stream(tuple(event.replace("Hello", " ") for event in events)))

    def test_converse_rejects_content_after_the_stop_event(self) -> None:
        with pytest.raises(AssertionError, match="event order"):
            assert_converse_stream(_stream((_CONVERSE[0], _CONVERSE[3], _CONVERSE[1], _CONVERSE[4])))
