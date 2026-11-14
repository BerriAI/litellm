"""
Regression tests for streaming connection pool leak fix.
"""

import asyncio
import json
from unittest.mock import MagicMock

import anyio
import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.aiohttp_transport import (
    AiohttpResponseStream,
    LiteLLMAiohttpTransport,
)


# ── aiohttp transport layer tests ──────────────────────────────


@pytest.mark.asyncio
async def test_aiohttp_transport_response_uses_stream_not_content():
    """handle_async_request must use stream= so aclose() propagates to AiohttpResponseStream."""

    class FakeSession:
        closed = False

        def __init__(self):
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        def request(self, **kwargs):
            class Resp:
                status = 200
                headers = {}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                @property
                def content(self):
                    class C:
                        async def iter_chunked(self, size):
                            yield b"data"

                    return C()

            return Resp()

    transport = LiteLLMAiohttpTransport(client=lambda: FakeSession())  # type: ignore
    response = await transport.handle_async_request(httpx.Request("GET", "http://example.com"))

    assert isinstance(response.stream, AiohttpResponseStream)


@pytest.mark.asyncio
async def test_aiohttp_response_stream_aclose_releases_connection():
    """AiohttpResponseStream.aclose() must call __aexit__ on the aiohttp response."""
    aexit_called = False

    class MockResponse:
        status = 200
        headers = {}

        @property
        def content(self):
            class C:
                async def iter_chunked(self, size):
                    yield b"data"

            return C()

        async def __aexit__(self, *args):
            nonlocal aexit_called
            aexit_called = True

    stream = AiohttpResponseStream(MockResponse())  # type: ignore
    await stream.aclose()
    assert aexit_called


# ── CustomStreamWrapper.aclose() tests ─────────────────────────


@pytest.mark.asyncio
async def test_aclose_falls_back_to_close():
    """OpenAI's AsyncStream has close() but not aclose(). Must fall back."""
    close_called = False

    class FakeAsyncStream:
        async def close(self):
            nonlocal close_called
            close_called = True

    wrapper = CustomStreamWrapper(
        completion_stream=FakeAsyncStream(),
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )

    await wrapper.aclose()
    assert close_called


@pytest.mark.asyncio
async def test_aclose_prefers_aclose_over_close():
    """When both aclose() and close() exist, aclose() should be preferred."""
    aclose_called = False
    close_called = False

    class FakeStream:
        async def aclose(self):
            nonlocal aclose_called
            aclose_called = True

        async def close(self):
            nonlocal close_called
            close_called = True

    wrapper = CustomStreamWrapper(
        completion_stream=FakeStream(),
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )

    await wrapper.aclose()
    assert aclose_called
    assert not close_called


@pytest.mark.asyncio
async def test_aclose_completes_under_cancellation():
    """aclose() must shield cleanup from CancelledError so streams actually close."""
    aclose_completed = False

    class SlowCloseStream:
        async def aclose(self):
            await anyio.sleep(0)
            nonlocal aclose_completed
            aclose_completed = True

    wrapper = CustomStreamWrapper(
        completion_stream=SlowCloseStream(),
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )

    with anyio.CancelScope() as scope:
        scope.cancel()
        await wrapper.aclose()

    assert aclose_completed


# ── Router stream_with_fallbacks cleanup tests ──────────────────


def _sse_chunk(content: str, finish_reason: str | None = None) -> bytes:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


_DONE = b"data: [DONE]\n\n"


class _TrackingStream(httpx.AsyncByteStream):
    """An upstream SSE body that records whether the router closed it."""

    def __init__(self, parts: list[bytes], fail: bool = False):
        self._parts = parts
        self._fail = fail
        self.closed = False

    async def __aiter__(self):
        for part in self._parts:
            yield part
        if self._fail:
            raise httpx.ReadError("upstream dropped the connection")

    async def aclose(self):
        self.closed = True


def _full_stream() -> _TrackingStream:
    return _TrackingStream([_sse_chunk("c1"), _sse_chunk("c2"), _sse_chunk("c3", "stop"), _DONE])


@pytest.fixture
def stream_router(monkeypatch: pytest.MonkeyPatch):
    from litellm.router import Router

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test",
                    "api_key": "fake",
                    "api_base": "https://primary.example/v1",
                },
            },
            {
                "model_name": "other-model",
                "litellm_params": {
                    "model": "openai/test",
                    "api_key": "fake",
                    "api_base": "https://fallback.example/v1",
                },
            },
        ],
        fallbacks=[{"test-model": ["other-model"]}],
    )
    litellm.in_memory_llm_clients_cache.flush_cache()


async def _open_stream(router, primary: _TrackingStream, fallback: _TrackingStream, mock_router):
    for host, body in (("primary", primary), ("fallback", fallback)):
        mock_router.post(f"https://{host}.example/v1/chat/completions").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
        )
    return await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}], stream=True)


@pytest.mark.asyncio
async def test_stream_with_fallbacks_closes_stream_on_generator_close(stream_router):
    """Closing the router's stream after one chunk (a client disconnect) must
    close the upstream response."""
    primary = _full_stream()

    with respx.mock(assert_all_called=False) as mock_router:
        result = await _open_stream(stream_router, primary, _full_stream(), mock_router)
        async for _ in result:
            break
        await result.aclose()

    assert primary.closed, "upstream stream was not closed when the client disconnected"


@pytest.mark.asyncio
async def test_stream_with_fallbacks_closes_stream_on_normal_completion(stream_router):
    """The upstream response must be closed even on normal completion."""
    primary = _full_stream()

    with respx.mock(assert_all_called=False) as mock_router:
        result = await _open_stream(stream_router, primary, _full_stream(), mock_router)
        async for _ in result:
            pass
        await result.aclose()

    assert primary.closed, "upstream stream was not closed after normal completion"


@pytest.mark.asyncio
async def test_stream_with_fallbacks_closes_both_on_fallback_disconnect(stream_router):
    """When a fallback is triggered and the client disconnects during fallback
    iteration, both the primary and the fallback upstream must be closed."""
    primary = _TrackingStream([], fail=True)
    fallback = _full_stream()

    with respx.mock() as mock_router:
        result = await _open_stream(stream_router, primary, fallback, mock_router)
        async for _ in result:
            break
        await result.aclose()

    assert primary.closed, "primary upstream stream was not closed"
    assert fallback.closed, "fallback upstream stream was not closed"
