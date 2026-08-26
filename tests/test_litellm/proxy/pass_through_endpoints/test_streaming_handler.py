"""Regression tests: chunk_processor must close the upstream httpx.Response
when the stream ends, including on client disconnect (generator aclose), so
the provider connection is released and backends like vLLM stop generating.
"""

from collections.abc import AsyncIterator, Coroutine
from datetime import datetime, timezone
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    AnthropicMessagesStreamHiddenParams,
    AnthropicMessagesStreamingResponse,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


class _TrackingStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.aclose_called = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.aclose_called = True


def _streaming_response(chunks: list[bytes]) -> tuple[httpx.Response, _TrackingStream]:
    stream: Final = _TrackingStream(chunks)
    response: Final = httpx.Response(
        200,
        stream=stream,
        request=httpx.Request("POST", "http://upstream.test/v1/messages"),
    )
    return response, stream


def _logging_obj_stub() -> MagicMock:
    logging_obj: Final = MagicMock()
    logging_obj.completion_start_time = None
    logging_obj.model_call_details = {}
    return logging_obj


def _chunk_processor(response: httpx.Response) -> AsyncIterator[bytes]:
    return PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-sonnet-4-5"},
        litellm_logging_obj=_logging_obj_stub(),
        endpoint_type=EndpointType.ANTHROPIC,
        start_time=datetime.now(timezone.utc),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/v1/messages",
    )


def _drop_scheduled_coroutine(async_coroutine: Coroutine[object, object, object]) -> None:
    async_coroutine.close()


@pytest.mark.asyncio
async def test_chunk_processor_closes_response_on_client_disconnect():
    response, stream = _streaming_response([b"event: message_start\n\n", b"event: message_stop\n\n"])
    generator: Final = _chunk_processor(response)

    with patch("litellm.proxy.pass_through_endpoints.streaming_handler.GLOBAL_LOGGING_WORKER") as worker:
        worker.ensure_initialized_and_enqueue.side_effect = _drop_scheduled_coroutine
        first_chunk: Final = await generator.__anext__()
        assert first_chunk == b"event: message_start\n\n"
        await generator.aclose()

    assert stream.aclose_called
    assert response.is_closed


@pytest.mark.asyncio
async def test_chunk_processor_closes_response_on_natural_completion():
    response, stream = _streaming_response([b"event: message_start\n\n", b"event: message_stop\n\n"])
    generator: Final = _chunk_processor(response)

    with patch("litellm.proxy.pass_through_endpoints.streaming_handler.GLOBAL_LOGGING_WORKER") as worker:
        worker.ensure_initialized_and_enqueue.side_effect = _drop_scheduled_coroutine
        collected: Final = [chunk async for chunk in generator]

    assert collected == [b"event: message_start\n\n", b"event: message_stop\n\n"]
    assert stream.aclose_called
    assert response.is_closed


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_response_aclose_closes_upstream_response():
    response, stream = _streaming_response([b"event: message_start\n\n"])
    wrapped: Final = AnthropicMessagesStreamingResponse(
        completion_stream=_chunk_processor(response),
        hidden_params=AnthropicMessagesStreamHiddenParams(additional_headers={}),
    )

    with patch("litellm.proxy.pass_through_endpoints.streaming_handler.GLOBAL_LOGGING_WORKER") as worker:
        worker.ensure_initialized_and_enqueue.side_effect = _drop_scheduled_coroutine
        first_chunk: Final = await wrapped.__anext__()
        assert first_chunk == b"event: message_start\n\n"
        await wrapped.aclose()

    assert stream.aclose_called
    assert response.is_closed
