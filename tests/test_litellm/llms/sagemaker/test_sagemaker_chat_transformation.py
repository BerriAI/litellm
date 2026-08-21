"""
Regression tests for LIT-4313: sagemaker_chat streaming must forward each AWS
event-stream frame as it arrives instead of buffering to a fixed 1024-byte
threshold and then draining a burst of deltas.

The buffering came from `response.iter_bytes(chunk_size=1024)` /
`response.aiter_bytes(chunk_size=1024)`: httpx's ByteChunker withholds bytes until
`chunk_size` accumulates, so the first client delta could not be produced until
enough later frames had arrived to cross 1024 bytes, inflating TTFT and turning a
steady provider stream into gap-then-burst delivery.
"""

import binascii
import json
import struct
from typing import AsyncIterator, Iterator
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig


def _encode_header(name: str, value: str) -> bytes:
    name_b = name.encode("utf-8")
    value_b = value.encode("utf-8")
    return struct.pack("B", len(name_b)) + name_b + struct.pack("B", 7) + struct.pack(">H", len(value_b)) + value_b


def _encode_event_frame(payload: bytes) -> bytes:
    """Encode one AWS event-stream message that botocore's EventStreamBuffer decodes."""
    headers = {
        ":event-type": "PayloadPart",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    headers_b = b"".join(_encode_header(k, v) for k, v in headers.items())
    total_len = 16 + len(headers_b) + len(payload)
    prelude = struct.pack(">I", total_len) + struct.pack(">I", len(headers_b))
    prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + prelude_crc + headers_b + payload
    message_crc = struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)
    return message + message_crc


def _delta_frame(index: int, content: str) -> bytes:
    sse = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
        )
        + "\n\n"
    )
    return _encode_event_frame(sse.encode("utf-8"))


def _make_frames(n: int) -> list[bytes]:
    # Small single-token frames (< 1024 bytes each) so a fixed 1024-byte chunker
    # would have to swallow several frames before releasing the first delta.
    frames = [_delta_frame(i, f"token{i} ") for i in range(n)]
    assert all(len(f) < 1024 for f in frames)
    return frames


class _CountingSyncStream(httpx.SyncByteStream):
    """Yields provider frames one at a time and records how many have been pulled."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames
        self.consumed = 0

    def __iter__(self) -> Iterator[bytes]:
        for frame in self._frames:
            self.consumed += 1
            yield frame


class _CountingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames
        self.consumed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            self.consumed += 1
            yield frame


class _FakeSyncClient:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def post(self, *args, **kwargs) -> httpx.Response:
        return self._response


class _FakeAsyncClient:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def post(self, *args, **kwargs) -> httpx.Response:
        return self._response


def _content_of(chunk) -> str | None:
    return chunk.choices[0].delta.content


def test_sync_first_event_emitted_after_a_single_frame():
    """The first delta must be available after exactly one source frame is pulled.

    With the old chunk_size=1024 the httpx chunker would consume several small
    frames before yielding, so `consumed` would be > 1 at the first delta.
    """
    frames = _make_frames(24)
    stream = _CountingSyncStream(frames)
    response = httpx.Response(200, stream=stream)

    wrapper = SagemakerChatConfig().get_sync_custom_stream_wrapper(
        model="phi-4",
        custom_llm_provider="sagemaker_chat",
        logging_obj=MagicMock(),
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data={},
        messages=[],
        client=_FakeSyncClient(response),
    )

    first = next(c for c in wrapper.completion_stream if c is not None and _content_of(c) is not None)
    assert _content_of(first) == "token0 "
    assert stream.consumed == 1


def test_sync_events_emitted_incrementally_without_bursting():
    """Each successive delta must correspond to exactly one newly-pulled frame."""
    frames = _make_frames(24)
    stream = _CountingSyncStream(frames)
    response = httpx.Response(200, stream=stream)

    wrapper = SagemakerChatConfig().get_sync_custom_stream_wrapper(
        model="phi-4",
        custom_llm_provider="sagemaker_chat",
        logging_obj=MagicMock(),
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data={},
        messages=[],
        client=_FakeSyncClient(response),
    )

    consumed_at_delta = [
        stream.consumed for chunk in wrapper.completion_stream if chunk is not None and _content_of(chunk) is not None
    ]

    assert consumed_at_delta == list(range(1, len(frames) + 1))


@pytest.mark.asyncio
async def test_async_first_event_emitted_after_a_single_frame():
    frames = _make_frames(24)
    stream = _CountingAsyncStream(frames)
    response = httpx.Response(200, stream=stream)

    wrapper = await SagemakerChatConfig().get_async_custom_stream_wrapper(
        model="phi-4",
        custom_llm_provider="sagemaker_chat",
        logging_obj=MagicMock(),
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data={},
        messages=[],
        client=_FakeAsyncClient(response),
    )

    consumed_at_delta = []
    async for chunk in wrapper.completion_stream:
        if chunk is not None and _content_of(chunk) is not None:
            consumed_at_delta.append(stream.consumed)

    assert consumed_at_delta == list(range(1, len(frames) + 1))


def test_signed_body_includes_stream_flag():
    """A streaming request must carry `stream: true` in the signed body sent to SageMaker.

    `stream` flows into the request body through the transformed request (`{**optional_params}`)
    and must survive SigV4 signing so the endpoint enables token-level streaming.
    """
    headers, signed_body = SagemakerChatConfig().sign_request(
        headers={},
        optional_params={
            "aws_access_key_id": "AKIATESTTESTTESTTEST",
            "aws_secret_access_key": "test-secret-key",
            "aws_region_name": "us-east-1",
        },
        request_data={"model": "phi-4", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        model="phi-4",
        stream=True,
    )
    assert signed_body is not None
    assert json.loads(signed_body)["stream"] is True


@pytest.mark.parametrize("split_size", [1, 3, 7, 64, 4096])
def test_decoder_reassembles_frames_across_arbitrary_byte_boundaries(split_size):
    """Correctness must not depend on chunk boundaries falling on frame edges.

    Removing `chunk_size=1024` lets httpx yield raw transport reads, so in
    production a single read can straddle several frames or split one frame in
    half. This re-chunks the concatenated stream at boundaries that deliberately
    ignore frame edges and asserts every delta still decodes, in order, exactly
    once - the guarantee botocore's EventStreamBuffer provides.
    """
    from litellm.llms.sagemaker.chat.transformation import AWSEventStreamDecoder

    frames = _make_frames(24)
    blob = b"".join(frames)
    chunks = [blob[i : i + split_size] for i in range(0, len(blob), split_size)]

    decoder = AWSEventStreamDecoder(model="phi-4", is_messages_api=True)
    texts = [
        _content_of(chunk)
        for chunk in decoder.iter_bytes(iter(chunks))
        if chunk is not None and _content_of(chunk) is not None
    ]

    assert texts == [f"token{i} " for i in range(len(frames))]


_INFERENCE_COMPONENT_HEADER = "X-Amzn-SageMaker-Inference-Component"

_STUB_COMPLETION_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "served-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _RequestCapturingHTTPHandler(HTTPHandler):
    """Injected transport that records exactly what sagemaker_chat put on the wire."""

    def __init__(self) -> None:
        super().__init__()
        self.request_headers: dict[str, str] = {}
        self.request_body: dict = {}

    def post(self, url: str, headers=None, data=None, **kwargs) -> httpx.Response:
        self.request_headers = dict(headers or {})
        self.request_body = json.loads(data)
        return httpx.Response(200, json=_STUB_COMPLETION_RESPONSE, request=httpx.Request("POST", url))


def _invoke_sagemaker_chat(monkeypatch, **extra_params) -> _RequestCapturingHTTPHandler:
    """Drive one sagemaker_chat completion against an injected transport.

    A Bedrock API key short-circuits SigV4 inside `BaseAWSLLM._sign_request`, which would hide
    whether the inference-component header is really covered by the signature, so it is cleared.
    """
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client = _RequestCapturingHTTPHandler()
    litellm.completion(
        model="sagemaker_chat/my-endpoint",
        messages=[{"role": "user", "content": "hi"}],
        aws_access_key_id="AKIATESTTESTTESTTEST",
        aws_secret_access_key="test-secret-key",
        aws_region_name="us-east-1",
        client=client,
        **extra_params,
    )
    return client


def test_model_id_is_sent_as_a_signed_inference_component_header(monkeypatch):
    """`model_id` names an inference component and must reach SageMaker as a signed header.

    Endpoints backed by inference components reject any request without
    `X-Amzn-SageMaker-Inference-Component` with HTTP 400 INFERENCE_COMPONENT_NAME_MISSING, so the
    header has to be built before `sign_request` runs and end up inside SignedHeaders.
    """
    client = _invoke_sagemaker_chat(monkeypatch, model_id="my-inference-component")

    assert client.request_headers[_INFERENCE_COMPONENT_HEADER] == "my-inference-component"
    assert "x-amzn-sagemaker-inference-component" in client.request_headers["Authorization"]


def test_no_inference_component_header_when_model_id_is_unset(monkeypatch):
    """Plain endpoints must not receive the header at all, not even an empty one."""
    client = _invoke_sagemaker_chat(monkeypatch)

    assert not any(name.lower() == _INFERENCE_COMPONENT_HEADER.lower() for name in client.request_headers)


def test_hf_model_name_becomes_the_body_model(monkeypatch):
    """`hf_model_name` names the served model, and containers that validate the body's `model`
    404 on the endpoint name, so it has to replace it rather than ride along as an extra field."""
    client = _invoke_sagemaker_chat(monkeypatch, hf_model_name="org/served-model")

    assert client.request_body["model"] == "org/served-model"
    assert "hf_model_name" not in client.request_body


def test_body_model_stays_the_endpoint_name_when_hf_model_name_is_unset(monkeypatch):
    """Without `hf_model_name` the body must keep the model it has today."""
    client = _invoke_sagemaker_chat(monkeypatch)

    assert client.request_body["model"] == "my-endpoint"
