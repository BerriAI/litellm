from collections.abc import AsyncIterator, Coroutine
from typing import Final, Protocol, runtime_checkable

import pytest

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.utils import ProxyLogging
from tests.test_litellm_rust.support.callback_contract import CallbackRoute
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec


@runtime_checkable
class ClosableStream(Protocol):
    def __aiter__(self) -> AsyncIterator[bytes]: ...

    async def __anext__(self) -> bytes: ...

    async def aclose(self) -> None: ...


async def release_stream_logging(coroutine: Coroutine[object, object, None]) -> None:
    GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(coroutine)


class TestStreamCallbackContract:
    @pytest.mark.asyncio
    async def test_unconsumed_stream_close_has_no_terminal_logging(
        self,
        callback_route: CallbackRoute,
        callback_server: RecordingServer,
        callback_stream_response: ResponseSpec,
    ) -> None:
        callback_server.enqueue(callback_stream_response)
        recorder: Final = RecordingLogger()
        logger: Final = callback_route.logger(recorder)
        stream: Final = await callback_route.invoke(callback_server, stream=True, litellm_logging_obj=logger)
        assert isinstance(stream, ClosableStream)
        await stream.aclose()
        await stream.aclose()
        await drain_logging()
        assert not any("success" in name or "failure" in name for name in recorder.names)
        assert len(callback_server.requests) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("deferred", (False, True), ids=("immediate", "deferred"))
    @pytest.mark.parametrize("ending", ("exhausted", "closed"))
    async def test_stream_terminal_logging_contract(
        self,
        callback_route: CallbackRoute,
        callback_server: RecordingServer,
        callback_stream_response: ResponseSpec,
        deferred: bool,
        ending: str,
    ) -> None:
        callback_server.enqueue(callback_stream_response)
        recorder: Final = RecordingLogger()
        logger: Final = callback_route.logger(recorder)
        if deferred:
            logger._on_deferred_stream_complete = release_stream_logging
        stream: Final = await callback_route.invoke(callback_server, stream=True, litellm_logging_obj=logger)
        assert isinstance(stream, ClosableStream)
        first: Final = await anext(stream)
        await drain_logging()
        assert first
        assert "async_log_success_event" not in recorder.names
        if ending == "exhausted":
            remaining: Final = tuple([chunk async for chunk in stream])
            assert first + b"".join(remaining) == b"".join(callback_stream_response.payloads())
        else:
            await stream.aclose()
        await drain_logging()
        assert recorder.names.count("async_log_success_event") == int(not deferred or ending == "closed")
        ProxyLogging._fire_deferred_stream_logging({"litellm_logging_obj": logger})
        await drain_logging()
        ProxyLogging._fire_deferred_stream_logging({"litellm_logging_obj": logger})
        await stream.aclose()
        await drain_logging()
        assert recorder.names.count("async_log_success_event") == 1
        assert not any("failure" in name for name in recorder.names)
        assert len(callback_server.requests) == 1
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("deferred", (False, True), ids=("immediate", "deferred"))
    async def test_stream_transport_failure_never_logs_success(
        self,
        callback_route: CallbackRoute,
        callback_server: RecordingServer,
        callback_stream_response: ResponseSpec,
        deferred: bool,
    ) -> None:
        callback_server.enqueue(
            ResponseSpec(body=None, events=callback_stream_response.events, disconnect_after_payloads=1)
        )
        recorder: Final = RecordingLogger()
        logger: Final = callback_route.logger(recorder)
        if deferred:
            logger._on_deferred_stream_complete = release_stream_logging
        stream: Final = await callback_route.invoke(callback_server, stream=True, litellm_logging_obj=logger)
        assert isinstance(stream, ClosableStream)
        with pytest.raises(litellm.APIConnectionError):
            async for _ in stream:
                pass
        await drain_logging()
        ProxyLogging._fire_deferred_stream_logging({"litellm_logging_obj": logger})
        await stream.aclose()
        await drain_logging()
        assert recorder.names.count("async_log_failure_event") == 1
        assert not any("success" in name for name in recorder.names)
        assert len(callback_server.requests) == 1
