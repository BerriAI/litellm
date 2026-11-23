import json
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
    decoder = AWSEventStreamDecoder(model="test-model")

    async def mock_iterator():
        yield _payload_part(b"\xff\xfe")

    chunks = [chunk async for chunk in decoder.aiter_bytes(mock_iterator()) if chunk is not None]

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_aiter_bytes_valid_chunk_followed_by_unicode_error():
    """
    Test that valid chunks are processed correctly even when followed by Unicode decode errors.
    This ensures errors don't corrupt or prevent processing of valid data that came before.

    Relevant issue: https://github.com/BerriAI/litellm/issues/9165
    """
    decoder = AWSEventStreamDecoder(model="test-model")

    async def mock_iterator():
        yield _payload_part(json.dumps({"token": {"text": "hello"}}))
        yield _payload_part(b"\xff\xfe")

    chunks = [chunk async for chunk in decoder.aiter_bytes(mock_iterator()) if chunk is not None]

    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello"


def _event_header(name: str, value: str) -> bytes:
    return struct.pack("!B", len(name)) + name.encode() + b"\x07" + struct.pack("!H", len(value)) + value.encode()


_PAYLOAD_PART_HEADERS: Final = (
    _event_header(":event-type", "PayloadPart")
    + _event_header(":content-type", "application/octet-stream")
    + _event_header(":message-type", "event")
)


def _frame(headers: bytes, body: str | bytes) -> bytes:
    encoded: Final = body.encode() if isinstance(body, str) else body
    prelude: Final = struct.pack("!II", 16 + len(headers) + len(encoded), len(headers))
    message: Final = prelude + struct.pack("!I", zlib.crc32(prelude)) + headers + encoded
    return message + struct.pack("!I", zlib.crc32(message))


def _payload_part(body: str | bytes) -> bytes:
    return _frame(_PAYLOAD_PART_HEADERS, body)


def _vllm_chunk(content: str) -> str:
    record: Final = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(record, ensure_ascii=False)}\n\n"


async def _stream_frames(
    frames: tuple[bytes, ...],
    is_messages_api: bool,
    mode: str,
    response_headers: Mapping[str, str] | None = None,
) -> AsyncIterator[GChunk | StreamingChatCompletionChunk | None]:
    decoder: Final = AWSEventStreamDecoder(model="m", is_messages_api=is_messages_api)
    if mode == "sync":
        for chunk in decoder.iter_bytes(iter(frames), response_headers=response_headers):
            yield chunk
        return

    async def stream():
        for frame in frames:
            yield frame

    async for chunk in decoder.aiter_bytes(stream(), response_headers=response_headers):
        yield chunk


_EVENT_STREAM_RESPONSE_HEADERS: Final = httpx.Headers(
    {"content-type": "application/vnd.amazon.eventstream", "x-amzn-RequestId": "req-sagemaker-1"}
)
_UNDECODABLE_SAGEMAKER_BODIES: Final = (
    pytest.param(b"", id="empty"),
    pytest.param(b"\x00\x00\x00\x05", id="shorter-than-a-prelude"),
    pytest.param(_payload_part(_vllm_chunk("Hél"))[:-8], id="truncated-first-message"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("is_messages_api", [True, False])
@pytest.mark.parametrize("body", _UNDECODABLE_SAGEMAKER_BODIES)
async def test_200_body_that_decodes_to_no_events_raises_bad_gateway_naming_the_request(
    body: bytes, is_messages_api: bool, mode: str
):
    stream: Final = _stream_frames((body,), is_messages_api, mode, _EVENT_STREAM_RESPONSE_HEADERS)

    with pytest.raises(SagemakerError) as exc_info:
        await anext(stream)

    assert exc_info.value.status_code == 502
    assert "HTTP 200" in exc_info.value.message
    assert "decoded to no events" in exc_info.value.message
    assert f"{len(body)} bytes received" in exc_info.value.message
    assert "application/vnd.amazon.eventstream" in exc_info.value.message
    assert "req-sagemaker-1" in exc_info.value.message
    assert exc_info.value.headers["x-amzn-requestid"] == "req-sagemaker-1"


_NOT_AN_EVENT_STREAM_BODIES: Final = (
    pytest.param(b"<html><body>Service Unavailable</body></html>", "text/html", id="html-page"),
    pytest.param(b'{"message":"Service Unavailable"}', "application/json", id="json-body"),
    pytest.param(b"\x00" * 16, "application/vnd.amazon.eventstream", id="zero-bytes"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("is_messages_api", [True, False])
@pytest.mark.parametrize(("body", "content_type"), _NOT_AN_EVENT_STREAM_BODIES)
async def test_200_body_that_is_not_an_event_stream_raises_bad_gateway_naming_the_request(
    body: bytes, content_type: str, is_messages_api: bool, mode: str
):
    response_headers: Final = httpx.Headers({"content-type": content_type, "x-amzn-RequestId": "req-sagemaker-1"})
    stream: Final = _stream_frames((body,), is_messages_api, mode, response_headers)

    with pytest.raises(SagemakerError) as exc_info:
        await anext(stream)

    assert exc_info.value.status_code == 502
    assert "stopped decoding as an AWS event stream after 0 events" in exc_info.value.message
    assert f"content-type='{content_type}'" in exc_info.value.message
    assert f"{len(body)} bytes received" in exc_info.value.message
    assert "Service Unavailable" not in exc_info.value.message
    assert exc_info.value.headers["x-amzn-requestid"] == "req-sagemaker-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_stream_whose_next_frame_is_not_an_event_raises_bad_gateway_after_the_complete_chunks(mode: str):
    stream: Final = _stream_frames(
        (_payload_part(_vllm_chunk("Hél")), b"<html><body>Service Unavailable</body></html>"),
        True,
        mode,
        _EVENT_STREAM_RESPONSE_HEADERS,
    )

    first: Final = await anext(stream)
    assert first is not None and first.choices[0].delta.content == "Hél"
    with pytest.raises(SagemakerError) as exc_info:
        await anext(stream)
    assert exc_info.value.status_code == 502
    assert "stopped decoding as an AWS event stream after 1 events" in exc_info.value.message
    assert "req-sagemaker-1" in exc_info.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_bad_gateway_for_a_first_frame_cut_before_its_checksum_keeps_its_text_out_of_the_error(mode: str):
    body: Final = _payload_part('data: {"token": {"text": "patient 4471 is allergic"}}\n\n')[:-4]
    stream: Final = _stream_frames((body,), False, mode, _EVENT_STREAM_RESPONSE_HEADERS)

    with pytest.raises(SagemakerError) as exc_info:
        await anext(stream)

    assert exc_info.value.status_code == 502
    assert "patient 4471" not in exc_info.value.message
    assert "req-sagemaker-1" in exc_info.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_stream_cut_inside_a_message_raises_bad_gateway_after_the_complete_chunks(mode: str):
    cut_message: Final = _payload_part(_vllm_chunk("lo"))[:-8]
    stream: Final = _stream_frames(
        (_payload_part(_vllm_chunk("Hél")), cut_message), True, mode, _EVENT_STREAM_RESPONSE_HEADERS
    )

    first: Final = await anext(stream)
    assert first is not None and first.choices[0].delta.content == "Hél"
    with pytest.raises(SagemakerError) as exc_info:
        await anext(stream)
    assert exc_info.value.status_code == 502
    assert f"{len(cut_message)} undecoded bytes after 1 events" in exc_info.value.message
    assert "req-sagemaker-1" in exc_info.value.message


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
