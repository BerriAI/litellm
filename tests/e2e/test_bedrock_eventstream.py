import base64
import json
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import MappingProxyType
from typing import Final

import pytest
from bedrock_eventstream import decode_bedrock_stream
from botocore.eventstream import ChecksumMismatch
from e2e_http import URL, NoBody, StreamingResponse, require_successful_call, send, streaming_outcome
from llm_translation.bedrock_stream import assert_converse_stream, assert_invoke_stream
from provider_diagnostics import NetworkFailureError, ProviderUnavailableError
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
    headers = MappingProxyType(
        {
            "content-type": "application/vnd.amazon.eventstream",
            "x-amzn-requestid": "stream-request",
            "x-litellm-call-id": "stream-call",
        }
    )

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
        with pytest.raises(ProviderUnavailableError) as caught:
            assert_converse_stream(result)
        assert caught.value.failure.status_code == 200
        assert caught.value.failure.evidence == "stream_event"
        assert caught.value.failure.request_id == "stream-request"
        assert caught.value.failure.call_id == "stream-call"

    @pytest.mark.parametrize("code", ["validationException", "throttlingException", "internalServerException"])
    def test_other_stream_exceptions_do_not_qualify_for_the_availability_retry(self, code: str) -> None:
        wire: Final = _frame('{"message":"provider rejected the request"}', code, "exception")
        result: Final = streaming_outcome(_BinaryResponse((wire,)), True, sent_at=0.0)
        with pytest.raises(AssertionError, match=code) as caught:
            assert_converse_stream(result)
        assert not isinstance(caught.value, ProviderUnavailableError)

    def test_corrupt_binary_frame_is_rejected(self) -> None:
        wire: Final = _frame('{"role":"assistant"}', "messageStart")
        with pytest.raises(ChecksumMismatch):
            tuple(decode_bedrock_stream((wire[:-1] + bytes([wire[-1] ^ 1]),)))

    @pytest.mark.parametrize("invoke", [False, True])
    @pytest.mark.parametrize("cut", [1, 7, 11, 12, 30, -1])
    def test_incomplete_frame_after_valid_completion_is_rejected(self, invoke: bool, cut: int) -> None:
        wire: Final = (
            b"".join(
                _frame(json.dumps({"bytes": base64.b64encode(event.encode()).decode()}), "chunk")
                for event in _INVOKE
            )
            if invoke
            else b"".join(
                _frame(json.dumps(value), key)
                for payload in _CONVERSE
                for key, value in _JSON_OBJECT.validate_json(payload).items()
            )
        )
        assertion: Final = assert_invoke_stream if invoke else assert_converse_stream
        assertion(streaming_outcome(_BinaryResponse((wire,)), True, sent_at=0.0))
        extra: Final = _frame('{"message":"unavailable"}', "serviceUnavailableException", "exception")[:cut]
        with pytest.raises(AssertionError, match="incomplete Bedrock frame"):
            assertion(streaming_outcome(_BinaryResponse((wire, extra)), True, sent_at=0.0))


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


class _InterruptedStream(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        _ = self.rfile.read(int(self.headers.get("content-length", "0")))
        binary: Final = self.path == "/binary"
        data: Final = (
            _frame('{"role":"assistant"}', "messageStart") if binary else b'data: {"text":"Hello"}\n\n'
        )
        self.send_response(200)
        self.send_header("content-type", "application/vnd.amazon.eventstream" if binary else "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.send_header("x-amzn-requestid", "interrupted-request")
        self.send_header("x-litellm-call-id", "interrupted-call")
        self.end_headers()
        _ = self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.mark.parametrize("path", ["binary", "sse"])
def test_interrupted_http_stream_preserves_network_classification_and_ids(path: str) -> None:
    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _InterruptedStream)
    worker: Final = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result: Final = send(
            URL(f"http://127.0.0.1:{server.server_port}/{path}"), headers=NoBody(), json=NoBody(), stream=True, timeout=5
        )
        assert result.status_code == 200 and not result.ok
        assert result.network_error is not None and result.network_error.kind == "network"
        with pytest.raises(NetworkFailureError, match="HTTP transfer failed") as caught:
            require_successful_call(result)
        assert caught.value.failure.provider is None
        assert caught.value.failure.request_id == "interrupted-request"
        assert caught.value.failure.call_id == "interrupted-call"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
