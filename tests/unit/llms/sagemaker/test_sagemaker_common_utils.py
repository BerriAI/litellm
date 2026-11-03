import json
import logging
import struct
import zlib
from collections.abc import AsyncIterator, Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from botocore.exceptions import ClientError, ConnectionClosedError

from litellm.llms.sagemaker.common_utils import AWSEventStreamDecoder, SagemakerError, client_error_details
from litellm.llms.sagemaker.completion.transformation import SagemakerConfig
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import StreamingChatCompletionChunk


# --------------------------------------------------------------------------- #
# get_sagemaker_response_stream_shape lazy-load tests                         #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_sagemaker_response_stream_shape_cache():
    """Prevent lru_cache leakage between tests in this module."""
    import litellm.llms.sagemaker.common_utils as mod

    mod.get_sagemaker_response_stream_shape.cache_clear()
    yield
    mod.get_sagemaker_response_stream_shape.cache_clear()


def test_sagemaker_response_stream_shape_lazy_loads_once():
    """
    get_sagemaker_response_stream_shape() loads from botocore at most once per process.
    """
    from unittest.mock import MagicMock, patch

    import litellm.llms.sagemaker.common_utils as mod

    sentinel = MagicMock()
    with patch.object(mod, "_load_sagemaker_response_stream_shape", return_value=sentinel) as mock_load:
        assert mod.get_sagemaker_response_stream_shape() is sentinel
        assert mod.get_sagemaker_response_stream_shape() is sentinel
        mock_load.assert_called_once()


def test_sagemaker_response_stream_shape_loaded_on_first_access():
    """
    get_sagemaker_response_stream_shape() loads once on first use.
    In a standard environment with botocore installed it must be non-None.
    """
    pytest.importorskip("botocore")
    from litellm.llms.sagemaker.common_utils import get_sagemaker_response_stream_shape

    assert get_sagemaker_response_stream_shape() is not None


def test_sagemaker_response_stream_shape_load_failure_returns_none():
    """
    If botocore's Loader raises (e.g. missing data files), _load_sagemaker_response_stream_shape
    should return None rather than propagating the exception, so the module
    still imports cleanly.
    """
    from unittest.mock import patch

    import litellm.llms.sagemaker.common_utils as mod

    pytest.importorskip("botocore")
    with patch(
        "botocore.loaders.Loader.load_service_model",
        side_effect=Exception("no data"),
    ):
        shape = mod._load_sagemaker_response_stream_shape()
        assert shape is None


@pytest.mark.parametrize("service_model", [["shapes"], None])
def test_sagemaker_response_stream_shape_is_none_for_a_service_model_that_is_not_a_mapping(
    service_model: object,
):
    pytest.importorskip("botocore")
    from unittest.mock import patch

    import litellm.llms.sagemaker.common_utils as mod

    with patch("botocore.loaders.Loader.load_service_model", return_value=service_model):
        assert mod._load_sagemaker_response_stream_shape() is None


def test_sagemaker_response_stream_shape_is_structure_shape():
    """
    The loaded shape should be the botocore StructureShape for
    InvokeEndpointWithResponseStreamOutput, not a plain dict or any other type.
    """
    pytest.importorskip("botocore")
    from botocore.model import StructureShape

    from litellm.llms.sagemaker.common_utils import get_sagemaker_response_stream_shape

    shape = get_sagemaker_response_stream_shape()
    assert shape is not None, "get_sagemaker_response_stream_shape() is None — botocore may not be installed"
    assert isinstance(shape, StructureShape)
    assert shape.name == "InvokeEndpointWithResponseStreamOutput"


def test_sagemaker_response_stream_shape_not_reloaded_on_new_decoder():
    """
    Creating multiple AWSEventStreamDecoder instances must not trigger
    additional botocore Loader calls — the shape is cached after first access.
    """
    from litellm.llms.sagemaker.common_utils import get_sagemaker_response_stream_shape

    decoder_a = AWSEventStreamDecoder.__new__(AWSEventStreamDecoder)
    decoder_b = AWSEventStreamDecoder.__new__(AWSEventStreamDecoder)

    assert "_response_stream_shape_cache" not in decoder_a.__dict__
    assert "_response_stream_shape_cache" not in decoder_b.__dict__

    first = get_sagemaker_response_stream_shape()
    second = get_sagemaker_response_stream_shape()
    assert first is second


def test_sagemaker_parse_message_from_event_raises_on_none_shape():
    """
    When get_sagemaker_response_stream_shape() returns None (botocore unavailable),
    _parse_message_from_event must raise SagemakerError before touching the
    botocore parser — not an opaque AttributeError from inside botocore.
    """
    from unittest.mock import MagicMock, patch

    import litellm.llms.sagemaker.common_utils as mod
    from litellm.llms.sagemaker.common_utils import SagemakerError

    decoder = AWSEventStreamDecoder.__new__(AWSEventStreamDecoder)
    decoder.model = "test-model"
    decoder.parser = MagicMock()
    decoder.content_blocks = []
    decoder.is_messages_api = None
    mock_event = MagicMock()

    with patch.object(mod, "get_sagemaker_response_stream_shape", return_value=None):
        with pytest.raises(SagemakerError) as exc_info:
            decoder._parse_message_from_event(mock_event)

    assert exc_info.value.status_code == 500
    assert "botocore" in str(exc_info.value.message).lower()
    # The botocore parser must never have been called
    mock_event.to_response_dict.assert_not_called()


@pytest.mark.asyncio
async def test_aiter_bytes_unicode_decode_error():
    """
    Test that AWSEventStreamDecoder.aiter_bytes() does not raise an error when encountering invalid UTF-8 bytes. (UnicodeDecodeError)


    Ensures stream processing continues despite the error.

    Relevant issue: https://github.com/BerriAI/litellm/issues/9165
    """
    # Create an instance of AWSEventStreamDecoder
    decoder = AWSEventStreamDecoder(model="test-model")

    # Create a mock event that will trigger a UnicodeDecodeError
    mock_event = MagicMock()
    mock_event.to_response_dict.return_value = {
        "status_code": 200,
        "headers": {},
        "body": b"\xff\xfe",  # Invalid UTF-8 bytes
    }

    # Create a mock EventStreamBuffer that yields our mock event
    mock_buffer = MagicMock()
    mock_buffer.__iter__.return_value = [mock_event]

    # Mock the EventStreamBuffer class
    with patch("botocore.eventstream.EventStreamBuffer", return_value=mock_buffer):
        # Create an async generator that yields some test bytes
        async def mock_iterator():
            yield b""

        # Process the stream
        chunks = []
        async for chunk in decoder.aiter_bytes(mock_iterator()):
            if chunk is not None:
                print("chunk=", chunk)
                chunks.append(chunk)

        # Verify that processing continued despite the error
        # The chunks list should be empty since we only sent invalid data
        assert len(chunks) == 0


@pytest.mark.asyncio
async def test_aiter_bytes_valid_chunk_followed_by_unicode_error():
    """
    Test that valid chunks are processed correctly even when followed by Unicode decode errors.
    This ensures errors don't corrupt or prevent processing of valid data that came before.

    Relevant issue: https://github.com/BerriAI/litellm/issues/9165
    """
    decoder = AWSEventStreamDecoder(model="test-model")

    # Create two mock events - first valid, then invalid
    mock_valid_event = MagicMock()
    mock_valid_event.to_response_dict.return_value = {
        "status_code": 200,
        "headers": {},
        "body": json.dumps({"token": {"text": "hello"}}).encode(),  # Valid data first
    }

    mock_invalid_event = MagicMock()
    mock_invalid_event.to_response_dict.return_value = {
        "status_code": 200,
        "headers": {},
        "body": b"\xff\xfe",  # Invalid UTF-8 bytes second
    }

    # Create a mock EventStreamBuffer that yields valid event first, then invalid
    mock_buffer = MagicMock()
    mock_buffer.__iter__.return_value = [mock_valid_event, mock_invalid_event]

    with patch("botocore.eventstream.EventStreamBuffer", return_value=mock_buffer):

        async def mock_iterator():
            yield b"test_bytes"

        chunks = []
        async for chunk in decoder.aiter_bytes(mock_iterator()):
            if chunk is not None:
                chunks.append(chunk)

        # Verify we got our valid chunk despite the subsequent error
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hello"  # Verify the content of the valid chunk


def _event_header(name: str, value: str) -> bytes:
    return struct.pack("!B", len(name)) + name.encode() + b"\x07" + struct.pack("!H", len(value)) + value.encode()


_PAYLOAD_PART_HEADERS: Final = (
    _event_header(":event-type", "PayloadPart")
    + _event_header(":content-type", "application/octet-stream")
    + _event_header(":message-type", "event")
)
_MODEL_STREAM_ERROR_HEADERS: Final = (
    _event_header(":exception-type", "ModelStreamError")
    + _event_header(":content-type", "application/json")
    + _event_header(":message-type", "exception")
)


def _frame(headers: bytes, body: str | bytes) -> bytes:
    encoded: Final = body.encode() if isinstance(body, str) else body
    prelude: Final = struct.pack("!II", 16 + len(headers) + len(encoded), len(headers))
    message: Final = prelude + struct.pack("!I", zlib.crc32(prelude)) + headers + encoded
    return message + struct.pack("!I", zlib.crc32(message))


def _payload_part(body: str | bytes) -> bytes:
    return _frame(_PAYLOAD_PART_HEADERS, body)


def _cut(stream: str, size: int) -> tuple[bytes, ...]:
    encoded: Final = stream.encode()
    return tuple(encoded[start : start + size] for start in range(0, len(encoded), size))


def _sse(record: Mapping[str, object]) -> str:
    return f"data: {json.dumps(record, ensure_ascii=False)}\n\n"


def _vllm_chunk(choices: list[Mapping[str, object]], **extra: object) -> str:
    return _sse(
        {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": choices, **extra}
    )


_VLLM_EVENTS: Final = (
    _vllm_chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]),
    _vllm_chunk([{"index": 0, "delta": {"content": "Hél"}, "finish_reason": None}]),
    _vllm_chunk([{"index": 0, "delta": {"content": "lo 👋"}, "finish_reason": "stop"}]),
    _vllm_chunk([], usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}),
    "data: [DONE]\n\n",
)
_VLLM_STREAM: Final = "".join(_VLLM_EVENTS)
_TGI_EVENTS: Final = tuple(
    f"data:{json.dumps({'index': 0, 'token': {'text': text}}, ensure_ascii=False)}\n\n" for text in ("Hél", "lo 👋")
)
_TGI_STREAM: Final = "".join(_TGI_EVENTS)
_TGI_KEEP_ALIVE_STREAM: Final = (
    f"{_TGI_EVENTS[0]}:\n\n: keep-alive\n\nevent: message\nid: 2\nretry: 1000\n{_TGI_EVENTS[1]}"
)
_LMI_STREAM: Final = "".join(
    f"{json.dumps({'token': {'text': text}}, ensure_ascii=False)}\n" for text in ("Hél", "lo 👋")
)


async def _stream_frames(
    frames: tuple[bytes, ...], is_messages_api: bool, mode: str
) -> AsyncIterator[GChunk | StreamingChatCompletionChunk | None]:
    decoder: Final = AWSEventStreamDecoder(model="m", is_messages_api=is_messages_api)
    if mode == "sync":
        for chunk in decoder.iter_bytes(iter(frames)):
            yield chunk
        return

    async def stream():
        for frame in frames:
            yield frame

    async for chunk in decoder.aiter_bytes(stream()):
        yield chunk


async def _decode(
    parts: tuple[str | bytes, ...], is_messages_api: bool, mode: str
) -> list[GChunk | StreamingChatCompletionChunk | None]:
    return [chunk async for chunk in _stream_frames(tuple(map(_payload_part, parts)), is_messages_api, mode)]


def _text(chunk: GChunk | StreamingChatCompletionChunk | None, is_messages_api: bool) -> str | None:
    if chunk is None:
        return None
    return chunk.choices[0].delta.content if is_messages_api else chunk["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "parts",
    [
        pytest.param(_VLLM_EVENTS, id="one-event-per-part"),
        pytest.param((*_VLLM_EVENTS[:3], _VLLM_EVENTS[3] + _VLLM_EVENTS[4]), id="usage-and-done-share-a-part"),
        pytest.param(
            (_VLLM_EVENTS[0] + _VLLM_EVENTS[1], _VLLM_EVENTS[2] + _VLLM_EVENTS[3], _VLLM_EVENTS[4]),
            id="event-pairs-share-a-part",
        ),
        pytest.param((*_VLLM_EVENTS[:4], "data: [DO", "NE]\n\n"), id="done-split-across-parts"),
        pytest.param((_VLLM_STREAM,), id="whole-stream-in-one-part"),
        pytest.param(_cut(_VLLM_STREAM, 7), id="seven-byte-parts"),
        pytest.param(_cut(_VLLM_STREAM, 1), id="one-byte-parts-split-every-utf8-character"),
    ],
)
async def test_messages_api_stream_keeps_every_chunk_however_sagemaker_cuts_the_parts(
    parts: tuple[str | bytes, ...], mode: str, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        chunks: Final = await _decode(parts, is_messages_api=True, mode=mode)

    assert None not in chunks
    assert "".join(chunk.choices[0].delta.content for chunk in chunks if chunk.choices) == "Héllo 👋"
    assert [(chunk.usage.prompt_tokens, chunk.usage.total_tokens) for chunk in chunks if chunk.usage] == [(7, 10)]
    assert caplog.records == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "parts",
    [
        pytest.param(_TGI_EVENTS, id="tgi-sse-one-event-per-part"),
        pytest.param((_TGI_STREAM,), id="tgi-sse-whole-stream-in-one-part"),
        pytest.param(_cut(_TGI_STREAM, 5), id="tgi-sse-five-byte-parts"),
        pytest.param(_cut(_TGI_STREAM, 1), id="tgi-sse-one-byte-parts-split-every-utf8-character"),
        pytest.param((_TGI_KEEP_ALIVE_STREAM,), id="tgi-sse-comments-and-fields-whole-stream-in-one-part"),
        pytest.param(_cut(_TGI_KEEP_ALIVE_STREAM, 1), id="tgi-sse-comments-and-fields-one-byte-parts"),
        pytest.param((_LMI_STREAM,), id="json-lines-whole-stream-in-one-part"),
        pytest.param(_cut(_LMI_STREAM, 5), id="json-lines-five-byte-parts"),
        pytest.param(tuple(_LMI_STREAM.splitlines()), id="undelimited-objects-one-per-part"),
    ],
)
async def test_completion_stream_keeps_every_token_however_sagemaker_cuts_the_parts(
    parts: tuple[str | bytes, ...], mode: str, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        chunks: Final = await _decode(parts, is_messages_api=False, mode=mode)

    assert None not in chunks
    assert "".join(chunk["text"] for chunk in chunks) == "Héllo 👋"
    assert caplog.records == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "last_part",
    [
        pytest.param(_VLLM_EVENTS[2][:20], id="cut-between-characters"),
        pytest.param(
            _VLLM_EVENTS[2].encode()[: _VLLM_EVENTS[2].encode().index("👋".encode()) + 2], id="cut-inside-a-character"
        ),
    ],
)
async def test_stream_that_ends_inside_a_record_raises_after_the_complete_ones(mode: str, last_part: str | bytes):
    stream: Final = _stream_frames(tuple(map(_payload_part, (*_VLLM_EVENTS[:2], last_part))), True, mode)

    assert [_text(await anext(stream), True) for _ in range(2)] == ["", "Hél"]
    with pytest.raises(SagemakerError, match="ended inside a record") as exc_info:
        await anext(stream)
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_text_left_over_that_is_not_a_record_is_logged_once(mode: str, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        chunks: Final = await _decode((*_VLLM_EVENTS[:3], "data: upstream hiccup\n\n"), is_messages_api=True, mode=mode)

    assert chunks[-1] is None
    assert "".join(chunk.choices[0].delta.content for chunk in chunks[:-1]) == "Héllo 👋"
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert "hiccup" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("parts", "is_messages_api", "status_code"),
    [
        pytest.param(
            (
                _VLLM_EVENTS[1],
                _sse(
                    {
                        "error": {
                            "message": "CUDA out of memory while decoding",
                            "type": "InternalServerError",
                            "code": 500,
                        }
                    }
                )
                + "data: [DONE]\n\n",
            ),
            True,
            500,
            id="vllm-500-error-event-shares-a-part-with-done",
        ),
        pytest.param(
            (
                _VLLM_EVENTS[1],
                _sse(
                    {"error": {"message": "CUDA out of memory while decoding", "type": "BadRequestError", "code": 400}}
                ),
                "data: [DONE]\n\n",
            ),
            True,
            500,
            id="vllm-400-error-event-in-its-own-part",
        ),
        pytest.param(
            (
                _VLLM_EVENTS[1],
                _sse(
                    {
                        "object": "error",
                        "message": "CUDA out of memory while decoding",
                        "type": "BadRequestError",
                        "code": 400,
                    }
                ),
            ),
            True,
            500,
            id="legacy-vllm-flat-error-event",
        ),
        pytest.param(
            (
                _VLLM_EVENTS[1],
                _sse({"error": {"message": "CUDA out of memory while decoding", "http_status_code": 422}}),
            ),
            True,
            422,
            id="tgi-messages-api-error-event",
        ),
        pytest.param(
            (
                _VLLM_EVENTS[1],
                _sse({"error": {"message": "CUDA out of memory while decoding", "http_status_code": 424}}),
            ),
            True,
            500,
            id="tgi-messages-api-generation-error-event",
        ),
        pytest.param(
            (
                _VLLM_EVENTS[1],
                _sse(
                    {"error": {"message": "CUDA out of memory while decoding", "type": "RateLimitError", "code": 429}}
                ),
            ),
            True,
            429,
            id="vllm-429-error-event",
        ),
        pytest.param(
            (_VLLM_EVENTS[1], _sse({"error": {"message": "CUDA out of memory while decoding", "code": "oom"}})),
            True,
            500,
            id="error-event-without-a-status",
        ),
        pytest.param(
            (
                _TGI_EVENTS[0]
                + f"data:{json.dumps({'error': 'CUDA out of memory while decoding', 'error_type': 'generation'})}\n\n",
            ),
            False,
            500,
            id="tgi-generation-error-event-shares-a-part-with-a-token",
        ),
        pytest.param(
            (
                _TGI_EVENTS[0],
                f"data:{json.dumps({'error': 'CUDA out of memory while decoding', 'error_type': 'overloaded'})}\n\n",
            ),
            False,
            429,
            id="tgi-overloaded-error-event",
        ),
        pytest.param(
            (
                _TGI_EVENTS[0],
                f"data:{json.dumps({'error': 'CUDA out of memory while decoding', 'error_type': 'incomplete_generation'})}\n\n",
            ),
            False,
            500,
            id="tgi-incomplete-generation-error-event",
        ),
    ],
)
async def test_container_error_event_raises_with_its_message_and_status_after_the_chunks_before_it(
    parts: tuple[str, ...], is_messages_api: bool, status_code: int, mode: str
):
    stream: Final = _stream_frames(tuple(map(_payload_part, parts)), is_messages_api, mode)

    assert _text(await anext(stream), is_messages_api) == "Hél"
    with pytest.raises(SagemakerError, match="CUDA out of memory while decoding") as exc_info:
        await anext(stream)
    assert exc_info.value.status_code == status_code


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("last_record", "message"),
    [
        pytest.param(
            {
                "token": {"id": 2, "text": "", "log_prob": 0.0, "error_msg": "CUDA out of memory while decoding"},
                "generated_text": "Hél",
                "details": {"finish_reason": "error"},
            },
            "SageMaker endpoint streamed a failed generation: CUDA out of memory while decoding",
            id="token-error-message",
        ),
        pytest.param(
            {
                "token": {"id": 2, "text": "", "log_prob": 0.0},
                "generated_text": "Hél",
                "details": {"finish_reason": "error"},
            },
            "SageMaker endpoint streamed a failed generation: no error message",
            id="error-finish-without-a-message",
        ),
    ],
)
async def test_lmi_failed_generation_raises_after_the_chunks_before_it(
    last_record: Mapping[str, object], message: str, mode: str
):
    lines: Final = "".join(
        f"{json.dumps(record, ensure_ascii=False)}\n"
        for record in ({"token": {"id": 1, "text": "Hél", "log_prob": -0.1}}, last_record)
    )
    stream: Final = _stream_frames((_payload_part(lines),), False, mode)

    assert _text(await anext(stream), False) == "Hél"
    with pytest.raises(SagemakerError) as exc_info:
        await anext(stream)
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == message


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_model_stream_error_raises_after_the_chunks_before_it(mode: str):
    frames: Final = (
        _payload_part(_VLLM_EVENTS[1]),
        _frame(
            _MODEL_STREAM_ERROR_HEADERS,
            json.dumps({"Message": "container exited", "ErrorCode": "ModelInvocationTimeExceeded"}),
        ),
        _payload_part(_VLLM_EVENTS[2]),
    )
    stream: Final = _stream_frames(frames, True, mode)

    assert _text(await anext(stream), True) == "Hél"
    with pytest.raises(SagemakerError) as exc_info:
        await anext(stream)
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == (
        "SageMaker stream failed with ModelStreamError (ModelInvocationTimeExceeded): container exited"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("parts", "is_messages_api"),
    [
        pytest.param(
            tuple(
                _vllm_chunk([{"index": 0, "delta": {"content": text}, "finish_reason": None}], error=None)
                for text in ("Hél", "lo 👋")
            ),
            True,
            id="vllm-chunks-with-null-error",
        ),
        pytest.param(
            tuple(
                f"data:{json.dumps({'index': 0, 'token': {'text': text}, 'error': None}, ensure_ascii=False)}\n\n"
                for text in ("Hél", "lo 👋")
            ),
            False,
            id="tgi-tokens-with-null-error",
        ),
    ],
)
async def test_chunk_with_a_null_error_field_is_delivered(
    parts: tuple[str, ...], is_messages_api: bool, mode: str, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        chunks: Final = await _decode(parts, is_messages_api=is_messages_api, mode=mode)

    assert "".join(_text(chunk, is_messages_api) or "" for chunk in chunks) == "Héllo 👋"
    assert caplog.records == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_record_that_is_not_a_chunk_is_logged_and_skipped(mode: str, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        chunks: Final = await _decode(
            (_VLLM_EVENTS[1], _sse({"object": "keepalive", "detail": "still decoding"}), _VLLM_EVENTS[2]),
            is_messages_api=True,
            mode=mode,
        )

    assert "".join(chunk.choices[0].delta.content for chunk in chunks) == "Héllo 👋"
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert "['detail', 'object']" in caplog.records[0].getMessage()
    assert "still decoding" not in caplog.records[0].getMessage()


class TestSagemakerTransform:
    def setup_method(self):
        self.config = SagemakerConfig()
        self.model = "test"
        self.logging_obj = MagicMock()

    def test_map_mistral_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "max_completion_tokens": 256,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_completion_tokens to max_tokens and override max_tokens
        assert result == {"temperature": 0.7, "max_new_tokens": 256}

    def test_mistral_max_tokens_backward_compat(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_tokens if max_completion_tokens is not provided
        assert result == {"temperature": 0.7, "max_new_tokens": 200}


NULL_MESSAGE_ERROR = ClientError(
    {"Error": {"Code": "InternalFailure", "Message": None}, "ResponseMetadata": {"HTTPStatusCode": 500}},
    "InvokeEndpoint",
)
DROPPED_CONNECTION_ERROR = ConnectionClosedError(endpoint_url="https://runtime.sagemaker.test/x")
HTTPX_STATUS_ERROR = httpx.HTTPStatusError(
    "boom", request=httpx.Request("POST", "https://runtime.sagemaker.test/x"), response=httpx.Response(400)
)
PLAIN_ERROR = RuntimeError("plain failure")


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_message"),
    [
        (
            ClientError(
                {
                    "Error": {"Code": "ValidationError", "Message": "bad input"},
                    "ResponseMetadata": {"HTTPStatusCode": 400},
                },
                "InvokeEndpoint",
            ),
            400,
            "bad input",
        ),
        (NULL_MESSAGE_ERROR, 500, str(NULL_MESSAGE_ERROR)),
        (DROPPED_CONNECTION_ERROR, 500, str(DROPPED_CONNECTION_ERROR)),
        (HTTPX_STATUS_ERROR, 500, str(HTTPX_STATUS_ERROR)),
        (PLAIN_ERROR, 500, str(PLAIN_ERROR)),
    ],
)
def test_client_error_details_reads_every_error_shape(error, expected_status, expected_message):
    assert client_error_details(error) == (expected_status, expected_message)
