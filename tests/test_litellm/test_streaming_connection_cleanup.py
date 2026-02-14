"""
Regression tests for streaming connection pool leak fix.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.aiohttp_transport import (
    AiohttpResponseStream,
    LiteLLMAiohttpTransport,
)


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
    response = await transport.handle_async_request(
        httpx.Request("GET", "http://example.com")
    )

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
            await asyncio.sleep(0)
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


@pytest.mark.asyncio
async def test_stream_with_fallbacks_closes_stream_on_generator_close():
    """Closing the generator from async_function_with_fallbacks must aclose() the stream."""
    from litellm.router import Router

    stream_closed = False

    class FakeStream:
        def __init__(self):
            self.chunks = ["chunk1", "chunk2", "chunk3"]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.chunks):
                raise StopAsyncIteration
            chunk = self.chunks[self.index]
            self.index += 1
            return chunk

        async def aclose(self):
            nonlocal stream_closed
            stream_closed = True

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test",
                    "api_key": "fake",
                },
            }
        ]
    )

    fake_stream = FakeStream()

    with patch.object(router, "acompletion", return_value=fake_stream):
        result = await router.async_function_with_fallbacks(
            original_function=router.acompletion,
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            num_retries=0,
        )

        async for chunk in result:
            break

        await result.aclose()

    assert stream_closed


def test_streaming_response_monkey_patch_applied():
    """_disconnect_aware_call must be applied to StreamingResponse.__call__."""
    from litellm.proxy.common_request_processing import _disconnect_aware_call
    from starlette.responses import StreamingResponse

    assert StreamingResponse.__call__ is _disconnect_aware_call
