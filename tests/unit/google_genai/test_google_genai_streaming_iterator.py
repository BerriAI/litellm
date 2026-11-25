import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.google_genai.streaming_iterator import (
    AsyncGoogleGenAIGenerateContentStreamingIterator,
    GoogleGenAIGenerateContentStreamingIterator,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.responses.streaming_iterator import drain_logging_tasks
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from tests.unit.litellm_core_utils.weakly_parked_logger import weakly_parked_logger


class _SuccessRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.logged: list[tuple[datetime, object]] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.logged.append((start_time, kwargs.get("completion_start_time")))


def _logging_obj(
    start_time: datetime = datetime(2026, 10, 9, 12, 0, 0), recorder: CustomLogger | None = None
) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="gemini-3.1-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="agenerate_content_stream",
        start_time=start_time,
        litellm_call_id="google-genai-stream",
        function_id="google-genai-stream",
        dynamic_async_success_callbacks=[recorder] if recorder else None,
    )
    logging_obj.update_environment_variables(
        model="gemini-3.1-flash",
        user="",
        optional_params={},
        litellm_params={"custom_llm_provider": "gemini"},
        custom_llm_provider="gemini",
    )
    return logging_obj


@pytest.mark.parametrize(
    "custom_llm_provider, expected_endpoint_type",
    [("gemini", EndpointType.GEMINI), ("vertex_ai", EndpointType.VERTEX_AI)],
)
@pytest.mark.parametrize(
    "iterator_cls",
    [
        AsyncGoogleGenAIGenerateContentStreamingIterator,
        GoogleGenAIGenerateContentStreamingIterator,
    ],
)
def test_streaming_logging_targets_the_provider_that_served_the_request(
    iterator_cls: (
        type[AsyncGoogleGenAIGenerateContentStreamingIterator] | type[GoogleGenAIGenerateContentStreamingIterator]
    ),
    custom_llm_provider: str,
    expected_endpoint_type: EndpointType,
):
    """Routing every google stream through the vertex handler bills gemini/* at vertex_ai/ rates."""
    iterator = iterator_cls(
        response=MagicMock(),
        model="gemini-3.1-flash-image",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider=custom_llm_provider,
    )

    assert iterator.endpoint_type is expected_endpoint_type


def _large_inline_data_event() -> str:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": "A" * 20000,
                            }
                        }
                    ]
                }
            }
        ]
    }
    return f"data: {json.dumps(payload)}"


def _sse_response(body: str) -> httpx.Response:
    return httpx.Response(200, stream=httpx.ByteStream(body.encode()))


@pytest.mark.asyncio
async def test_async_streaming_iterator_yields_complete_sse_events():
    """Large inlineData must not be split across byte-chunk boundaries."""
    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=_sse_response(_large_inline_data_event() + "\n\n"),
        model="gemini-3.1-flash-image-preview",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = await iterator.__anext__()
    assert chunk.startswith(b"data: ")
    assert chunk.endswith(b"\n\n")
    assert (
        json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][0]["inlineData"]["mimeType"]
        == "image/jpeg"
    )


def test_sync_streaming_iterator_yields_complete_sse_events():
    iterator = GoogleGenAIGenerateContentStreamingIterator(
        response=_sse_response(_large_inline_data_event() + "\n\n"),
        model="gemini-3.1-flash-image-preview",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = next(iterator)
    assert chunk.startswith(b"data: ")
    assert chunk.endswith(b"\n\n")
    assert json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][0]["inlineData"][
        "data"
    ].startswith("A")


@pytest.mark.asyncio
async def test_async_streaming_iterator_preserves_multi_field_sse_event():
    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=_sse_response('event: message\ndata: {"text":"hi"}\n\n'),
        model="gemini-test",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = await iterator.__anext__()
    assert chunk == b'event: message\ndata: {"text":"hi"}\n\n'


@pytest.mark.asyncio
async def test_async_streaming_iterator_forwards_sse_comment_events():
    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=_sse_response(": keepalive\n\n"),
        model="gemini-test",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = await iterator.__anext__()
    assert chunk == b": keepalive\n\n"


_LINE_SEPARATED_TEXT: Final = "before\u2028between\u2029after"


async def _first_chunk(
    iterator: GoogleGenAIGenerateContentStreamingIterator | AsyncGoogleGenAIGenerateContentStreamingIterator,
) -> bytes:
    if isinstance(iterator, AsyncGoogleGenAIGenerateContentStreamingIterator):
        return await iterator.__anext__()
    return next(iterator)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "iterator_cls",
    [AsyncGoogleGenAIGenerateContentStreamingIterator, GoogleGenAIGenerateContentStreamingIterator],
)
async def test_streaming_iterator_keeps_unicode_line_separators_inside_the_event(
    iterator_cls: (
        type[AsyncGoogleGenAIGenerateContentStreamingIterator] | type[GoogleGenAIGenerateContentStreamingIterator]
    ),
) -> None:
    payload: Final = {"candidates": [{"content": {"parts": [{"text": _LINE_SEPARATED_TEXT}]}}]}
    event: Final = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    iterator: Final = iterator_cls(
        response=_sse_response(event),
        model="gemini-3.8-flash",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk: Final = await _first_chunk(iterator)

    assert chunk == event.encode()
    assert (
        json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][0]["text"] == _LINE_SEPARATED_TEXT
    )


def _gemini_event(text: str, finish_reason: str | None = None) -> str:
    candidate: Final = {"content": {"role": "model", "parts": [{"text": text}]}, "index": 0}
    payload: Final = {
        "candidates": [{**candidate, "finishReason": finish_reason} if finish_reason else candidate],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
        "modelVersion": "gemini-3.1-flash",
    }
    return f"data: {json.dumps(payload)}\n\n"


@pytest.mark.asyncio
async def test_async_stream_logs_its_first_chunk_time_measured_from_the_call_start():
    call_start: Final = datetime(2026, 10, 9, 12, 0, 0)
    recorder: Final = _SuccessRecorder()
    logging_obj: Final = _logging_obj(call_start, recorder)
    response: Final = httpx.Response(200, content=(_gemini_event("Hel") + _gemini_event("lo", "STOP")).encode())
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=response,
        model="gemini-3.1-flash",
        logging_obj=logging_obj,
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )
    tasks_before: Final = asyncio.all_tasks()

    _ = await iterator.__anext__()
    first_chunk_at: Final = logging_obj.completion_start_time
    _ = [chunk async for chunk in iterator]
    await asyncio.gather(*(asyncio.all_tasks() - tasks_before))

    assert first_chunk_at is not None
    assert recorder.logged == [(call_start, first_chunk_at)]


@pytest.mark.asyncio
async def test_async_stream_success_log_survives_garbage_collection_while_its_callback_is_pending():
    async with weakly_parked_logger() as logger:
        response: Final = httpx.Response(200, content=(_gemini_event("Hel") + _gemini_event("lo", "STOP")).encode())
        iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
            response=response,
            model="gemini-3.1-flash",
            logging_obj=_logging_obj(recorder=logger),
            generate_content_provider_config=MagicMock(),
            litellm_metadata={},
            custom_llm_provider="gemini",
        )

        yielded: Final = len([chunk async for chunk in iterator])
        survived: Final = await logger.survives_collection()
        await drain_logging_tasks()

        assert (yielded, survived, logger.parks, logger.finished.is_set()) == (2, True, 1, True)


class _ControlledAsyncStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        wait_after_chunks: asyncio.Event | None = None,
        error_after_chunks: Exception | None = None,
    ) -> None:
        self._chunks: Final = chunks
        self._wait_after_chunks: Final = wait_after_chunks
        self._error_after_chunks: Final = error_after_chunks
        self.closed: bool = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._error_after_chunks is not None:
            raise self._error_after_chunks
        if self._wait_after_chunks is not None:
            await self._wait_after_chunks.wait()

    async def aclose(self) -> None:
        self.closed = True


class _FakeProxyLogging:
    def __init__(self) -> None:
        self.disconnect_slot_releases: int = 0

    async def arelease_max_parallel_requests_on_disconnect(self, user_api_key_dict: UserAPIKeyAuth) -> None:
        self.disconnect_slot_releases += 1


def _provider_logging_obj(custom_llm_provider: str, recorder: CustomLogger) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="gemini-3.1-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="agenerate_content_stream",
        start_time=datetime(2026, 10, 9, 12, 0, 0),
        litellm_call_id="google-genai-cancel-stream",
        function_id="google-genai-cancel-stream",
        dynamic_async_success_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        model="gemini-3.1-flash",
        user="",
        optional_params={},
        litellm_params={"custom_llm_provider": custom_llm_provider},
        custom_llm_provider=custom_llm_provider,
    )
    return logging_obj


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai"])
async def test_async_stream_logs_once_when_cancelled_after_last_chunk(custom_llm_provider: str) -> None:
    recorder: Final = _SuccessRecorder()
    wait_after_chunks: Final = asyncio.Event()
    stream: Final = _ControlledAsyncStream(
        (_gemini_event("Hel").encode(), _gemini_event("lo", "STOP").encode()),
        wait_after_chunks=wait_after_chunks,
    )
    response: Final = httpx.Response(200, stream=stream)
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=response,
        model="gemini-3.1-flash",
        logging_obj=_provider_logging_obj(custom_llm_provider, recorder),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider=custom_llm_provider,
    )

    first: Final = await iterator.__anext__()
    second: Final = await iterator.__anext__()
    assert first.startswith(b"data: ")
    assert second.startswith(b"data: ")

    trailing_read: Final = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0)
    trailing_read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await trailing_read

    await drain_logging_tasks()
    assert len(recorder.logged) == 1

    await iterator.aclose()
    await drain_logging_tasks()
    assert len(recorder.logged) == 1
    assert response.is_closed is True


@pytest.mark.asyncio
async def test_async_stream_skips_success_log_when_cancelled_before_any_chunk() -> None:
    recorder: Final = _SuccessRecorder()
    wait_after_chunks: Final = asyncio.Event()
    response: Final = httpx.Response(200, stream=_ControlledAsyncStream((), wait_after_chunks=wait_after_chunks))
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=response,
        model="gemini-3.1-flash",
        logging_obj=_provider_logging_obj("gemini", recorder),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    first_read: Final = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0)
    first_read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_read

    await drain_logging_tasks()
    await iterator.aclose()
    assert recorder.logged == []
    assert response.is_closed is True


@pytest.mark.asyncio
async def test_async_stream_aclose_logs_collected_chunks_once_without_trailing_anext() -> None:
    recorder: Final = _SuccessRecorder()
    wait_after_chunks: Final = asyncio.Event()
    response: Final = httpx.Response(
        200,
        stream=_ControlledAsyncStream(
            (_gemini_event("Hel").encode(), _gemini_event("lo", "STOP").encode()),
            wait_after_chunks=wait_after_chunks,
        ),
    )
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=response,
        model="gemini-3.1-flash",
        logging_obj=_provider_logging_obj("vertex_ai", recorder),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="vertex_ai",
    )

    _ = await iterator.__anext__()
    _ = await iterator.__anext__()

    await iterator.aclose()
    await drain_logging_tasks()
    await iterator.aclose()
    await drain_logging_tasks()

    assert len(recorder.logged) == 1
    assert response.is_closed is True


@pytest.mark.asyncio
async def test_async_stream_normal_completion_and_aclose_logs_once() -> None:
    recorder: Final = _SuccessRecorder()
    response: Final = httpx.Response(200, content=(_gemini_event("Hel") + _gemini_event("lo", "STOP")).encode())
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=response,
        model="gemini-3.1-flash",
        logging_obj=_provider_logging_obj("gemini", recorder),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunks: Final = [chunk async for chunk in iterator]
    await iterator.aclose()
    await drain_logging_tasks()

    assert len(chunks) == 2
    assert len(recorder.logged) == 1
    assert response.is_closed is True


@pytest.mark.asyncio
async def test_async_stream_error_mid_stream_does_not_log_success_on_aclose() -> None:
    recorder: Final = _SuccessRecorder()
    response: Final = httpx.Response(
        200,
        stream=_ControlledAsyncStream(
            (_gemini_event("Hel").encode(),),
            error_after_chunks=httpx.ReadError("upstream reset"),
        ),
    )
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=response,
        model="gemini-3.1-flash",
        logging_obj=_provider_logging_obj("gemini", recorder),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    first: Final = await iterator.__anext__()
    assert first.startswith(b"data: ")
    with pytest.raises(httpx.ReadError, match="upstream reset"):
        await iterator.__anext__()

    await iterator.aclose()
    await drain_logging_tasks()
    assert recorder.logged == []
    assert response.is_closed is True


@pytest.mark.asyncio
async def test_proxy_disconnect_cleanup_logs_collected_chunks_without_double_slot_release() -> None:
    recorder: Final = _SuccessRecorder()
    logging_obj: Final = _provider_logging_obj("vertex_ai", recorder)
    wait_after_chunks: Final = asyncio.Event()
    response: Final = httpx.Response(
        200,
        stream=_ControlledAsyncStream(
            (_gemini_event("Hel").encode(), _gemini_event("lo", "STOP").encode()),
            wait_after_chunks=wait_after_chunks,
        ),
    )
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=response,
        model="gemini-3.1-flash",
        logging_obj=logging_obj,
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="vertex_ai",
    )
    proxy_logging: Final = _FakeProxyLogging()

    _ = await iterator.__anext__()
    _ = await iterator.__anext__()

    await ProxyBaseLLMRequestProcessing.finalize_streaming_generator_cleanup(
        request=None,
        request_data={"litellm_logging_obj": logging_obj},
        response=iterator,
        stream_completed=False,
        client_disconnected=True,
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        proxy_logging_obj=proxy_logging,  # pyright: ignore[reportArgumentType]  # lightweight disconnect-slot test double
    )
    await drain_logging_tasks()

    assert len(recorder.logged) == 1
    assert response.is_closed is True
    assert proxy_logging.disconnect_slot_releases == 0


@pytest.mark.asyncio
async def test_proxy_disconnect_cleanup_without_chunks_releases_slot() -> None:
    recorder: Final = _SuccessRecorder()
    logging_obj: Final = _provider_logging_obj("gemini", recorder)
    wait_after_chunks: Final = asyncio.Event()
    response: Final = httpx.Response(200, stream=_ControlledAsyncStream((), wait_after_chunks=wait_after_chunks))
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=response,
        model="gemini-3.1-flash",
        logging_obj=logging_obj,
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )
    proxy_logging: Final = _FakeProxyLogging()

    await ProxyBaseLLMRequestProcessing.finalize_streaming_generator_cleanup(
        request=None,
        request_data={"litellm_logging_obj": logging_obj},
        response=iterator,
        stream_completed=False,
        client_disconnected=True,
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        proxy_logging_obj=proxy_logging,  # pyright: ignore[reportArgumentType]  # lightweight disconnect-slot test double
    )
    await drain_logging_tasks()

    assert recorder.logged == []
    assert response.is_closed is True
    assert proxy_logging.disconnect_slot_releases == 1
