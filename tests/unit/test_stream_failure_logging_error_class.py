import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import litellm
from litellm.exceptions import InternalServerError, MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup


class _MockSSEHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        sse_payload = (
            'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}\n\n'
        )
        self.wfile.write(sse_payload.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def mock_sse_server():
    server = HTTPServer(("127.0.0.1", 0), _MockSSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def test_get_error_information_unwraps_mid_stream_fallback_error():
    inner = InternalServerError(
        message="Anthropic overloaded",
        response=None,
        llm_provider="anthropic",
        model="claude-sonnet-4-5",
    )
    wrapper = MidStreamFallbackError(
        message="MidStreamFallbackError: overloaded",
        original_exception=inner,
        model="claude-sonnet-4-5",
        llm_provider="anthropic",
    )

    info = StandardLoggingPayloadSetup.get_error_information(wrapper)
    assert info["error_class"] == "InternalServerError"
    assert info["error_code"] == "500"
    assert info["llm_provider"] == "anthropic"


def test_stream_failure_callback_logs_mapped_error_class_sync(mock_sse_server):
    logged = []

    class _SyncFailLogger(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            std_obj = kwargs.get("standard_logging_object", {})
            if "error_information" in std_obj:
                logged.append(std_obj["error_information"])

    original_callbacks = list(litellm.callbacks)
    litellm.callbacks = [_SyncFailLogger()]
    try:
        with pytest.raises(Exception) as exc_info:
            for _ in litellm.completion(
                model="anthropic/claude-sonnet-4-5",
                api_key="sk-mock-key",
                api_base=f"http://127.0.0.1:{mock_sse_server.server_address[1]}",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ):
                pass

        # Underlying exception raised/wrapped should be InternalServerError
        raised = getattr(exc_info.value, "original_exception", None) or exc_info.value
        assert isinstance(raised, InternalServerError)

        for _ in range(50):
            if logged:
                break
            time.sleep(0.05)

        assert len(logged) > 0
        info = logged[0]
        assert info["error_class"] == "InternalServerError"
        assert str(info["error_code"]) == "500"
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_stream_failure_callback_logs_mapped_error_class_async(mock_sse_server):
    logged = []

    class _AsyncFailLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            std_obj = kwargs.get("standard_logging_object", {})
            if "error_information" in std_obj:
                logged.append(std_obj["error_information"])

    original_callbacks = list(litellm.callbacks)
    litellm.callbacks = [_AsyncFailLogger()]
    try:
        with pytest.raises(Exception) as exc_info:
            stream = await litellm.acompletion(
                model="anthropic/claude-sonnet-4-5",
                api_key="sk-mock-key",
                api_base=f"http://127.0.0.1:{mock_sse_server.server_address[1]}",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for _ in stream:
                pass

        raised = getattr(exc_info.value, "original_exception", None) or exc_info.value
        assert isinstance(raised, InternalServerError)

        for _ in range(50):
            if logged:
                break
            await asyncio.sleep(0.05)

        assert len(logged) > 0
        info = logged[0]
        assert info["error_class"] == "InternalServerError"
        assert str(info["error_code"]) == "500"
    finally:
        litellm.callbacks = original_callbacks
