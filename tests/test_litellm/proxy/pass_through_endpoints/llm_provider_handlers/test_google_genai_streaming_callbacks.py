"""
Regression tests for GitHub issue #24097:
  success_callback functions silently skipped for /models/{model}:streamGenerateContent

Root cause: streaming_iterator tagged collected chunks as VERTEX_AI, so
_route_streaming_logging_to_handler had no matching branch and silently skipped
every callback. Sync callers also lost callbacks because __next__ never invoked
the logging route on StopIteration at all.

Run:
    poetry run python -m pytest tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_google_genai_streaming_callbacks.py -v
"""

import asyncio
from datetime import datetime
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Enum: GOOGLE_GENAI must exist
# ---------------------------------------------------------------------------


class TestEndpointTypeEnum:
    def test_google_genai_member_exists(self):
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            EndpointType,
        )

        assert hasattr(
            EndpointType, "GOOGLE_GENAI"
        ), "EndpointType.GOOGLE_GENAI must exist for streaming callback routing"
        assert EndpointType.GOOGLE_GENAI == "google-genai"

    def test_existing_endpoint_types_preserved(self):
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            EndpointType,
        )

        # Sanity: adding GOOGLE_GENAI must not break existing members
        for name in ("VERTEX_AI", "ANTHROPIC", "OPENAI", "GENERIC"):
            assert hasattr(EndpointType, name), f"EndpointType.{name} regressed"


# ---------------------------------------------------------------------------
# 2. Async iterator: tags chunks as GOOGLE_GENAI and uses real URL with model
#
# Runtime behavior test — drives one iteration end-to-end, captures the kwargs
# passed to _route_streaming_logging_to_handler. Avoids inspect.getsource which
# would falsely fail on a comment that mentions VERTEX_AI.
# ---------------------------------------------------------------------------


class _FakeAsyncBytesResponse:
    """Minimal async-iterable response stub that yields a single chunk then ends."""

    def __init__(self, chunks: List[bytes]):
        self._chunks = chunks

    def aiter_bytes(self):
        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()


@pytest.mark.asyncio
async def test_async_iterator_routes_with_google_genai_endpoint_type():
    """The async iterator must call _route_streaming_logging_to_handler with
    endpoint_type=GOOGLE_GENAI and url_route containing the real model name."""
    from litellm.google_genai.streaming_iterator import (
        AsyncGoogleGenAIGenerateContentStreamingIterator,
    )
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        EndpointType,
    )

    captured: dict = {}

    async def _capture(*args, **kwargs):
        captured.update(kwargs)

    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler"
        ".PassThroughStreamingHandler._route_streaming_logging_to_handler",
        side_effect=_capture,
    ):
        iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
            response=_FakeAsyncBytesResponse([b"data: {}\n"]),
            model="gemini-2.5-pro",
            logging_obj=MagicMock(),
            generate_content_provider_config=MagicMock(),
            litellm_metadata={},
            custom_llm_provider="gemini",
            request_body={},
        )
        # Drive the iterator to completion so __anext__ catches StopAsyncIteration
        # and triggers _handle_async_streaming_logging.
        chunks = []
        async for chunk in iterator:
            chunks.append(chunk)

        # _handle_async_streaming_logging schedules via asyncio.create_task; let it run.
        await asyncio.sleep(0)

    assert captured, (
        "_route_streaming_logging_to_handler was never called — "
        "async iterator failed to invoke logging on stream end (issue #24097)"
    )
    assert captured["endpoint_type"] == EndpointType.GOOGLE_GENAI, (
        f"Expected GOOGLE_GENAI, got {captured['endpoint_type']!r} — "
        "iterator is misclassifying chunks (the original PR #24114 bug)"
    )
    assert "gemini-2.5-pro" in captured["url_route"], (
        f"Expected url_route to include the model name, got {captured['url_route']!r} — "
        "downstream URL parsing in GeminiPassthroughLoggingHandler will fail"
    )
    assert (
        captured["model"] == "gemini-2.5-pro"
    ), f"Expected explicit model kwarg, got {captured.get('model')!r}"


# ---------------------------------------------------------------------------
# 3. Sync iterator: same routing + model, fixes the gap PR #24114 missed
# ---------------------------------------------------------------------------


class _FakeSyncBytesResponse:
    """Minimal sync-iterable response stub."""

    def __init__(self, chunks: List[bytes]):
        self._chunks = chunks

    def iter_bytes(self):
        return iter(self._chunks)


def test_sync_iterator_routes_with_google_genai_endpoint_type():
    """The sync iterator must also invoke logging on StopIteration, with the
    same GOOGLE_GENAI tagging and url_route as the async path. This was the
    silent gap left over from PR #24114, which only patched the async side."""
    from litellm.google_genai.streaming_iterator import (
        GoogleGenAIGenerateContentStreamingIterator,
    )
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        EndpointType,
    )

    captured: dict = {}

    async def _capture(*args, **kwargs):
        captured.update(kwargs)

    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler"
        ".PassThroughStreamingHandler._route_streaming_logging_to_handler",
        side_effect=_capture,
    ):
        iterator = GoogleGenAIGenerateContentStreamingIterator(
            response=_FakeSyncBytesResponse([b"data: {}\n"]),
            model="gemini-2.5-pro",
            logging_obj=MagicMock(),
            generate_content_provider_config=MagicMock(),
            litellm_metadata={},
            custom_llm_provider="gemini",
            request_body={},
        )
        # Drive to completion so __next__ catches StopIteration.
        chunks = list(iterator)
        assert len(chunks) == 1

    assert captured, (
        "_route_streaming_logging_to_handler was never called from sync "
        "iterator — sync callers still silently lose callbacks (PR #24114 gap)"
    )
    assert captured["endpoint_type"] == EndpointType.GOOGLE_GENAI
    assert "gemini-2.5-pro" in captured["url_route"]
    assert captured["model"] == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# 4. Streaming handler: GOOGLE_GENAI is routed to GeminiPassthroughLoggingHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_handler_routes_google_genai_to_gemini_handler():
    """_route_streaming_logging_to_handler with endpoint_type=GOOGLE_GENAI must
    dispatch to GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks.
    """
    from litellm.proxy.pass_through_endpoints.streaming_handler import (
        PassThroughStreamingHandler,
    )
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        EndpointType,
    )

    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = AsyncMock()
    mock_logging_obj._should_run_sync_callbacks_for_async_calls = MagicMock(
        return_value=False
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler"
        ".GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks",
        return_value={"result": MagicMock(), "kwargs": {}},
    ) as mock_gemini:
        await PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/models/gemini-2.5-pro:streamGenerateContent",
            request_body={},
            endpoint_type=EndpointType.GOOGLE_GENAI,
            start_time=datetime.now(),
            raw_bytes=[b"data: {}\n"],
            end_time=datetime.now(),
            model="gemini-2.5-pro",
        )

    assert mock_gemini.call_count == 1, (
        "GeminiPassthroughLoggingHandler was NOT called for GOOGLE_GENAI — "
        "callbacks would be silently skipped (issue #24097 not fixed)"
    )
    # Verify the model kwarg is forwarded so the handler does not have to
    # fall back to URL parsing (which returns 'unknown' if the URL pattern shifts).
    call_kwargs = mock_gemini.call_args.kwargs
    assert call_kwargs.get("model") == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_streaming_handler_does_not_route_vertex_ai_to_gemini_handler():
    """Regression guard: VERTEX_AI must continue to dispatch to
    VertexPassthroughLoggingHandler, not the new Gemini one."""
    from litellm.proxy.pass_through_endpoints.streaming_handler import (
        PassThroughStreamingHandler,
    )
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        EndpointType,
    )

    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = AsyncMock()
    mock_logging_obj._should_run_sync_callbacks_for_async_calls = MagicMock(
        return_value=False
    )

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.streaming_handler"
            ".GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks",
        ) as mock_gemini,
        patch(
            "litellm.proxy.pass_through_endpoints.streaming_handler"
            ".VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks",
            return_value={"result": MagicMock(), "kwargs": {}},
        ) as mock_vertex,
    ):
        await PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/v1beta/projects/p/locations/l/publishers/google/models/gemini:streamGenerateContent",
            request_body={},
            endpoint_type=EndpointType.VERTEX_AI,
            start_time=datetime.now(),
            raw_bytes=[b"data: {}\n"],
            end_time=datetime.now(),
            model="gemini-2.5-pro",
        )

    assert mock_gemini.call_count == 0, (
        "GeminiPassthroughLoggingHandler was called for VERTEX_AI — "
        "routing regression detected"
    )
    assert (
        mock_vertex.call_count == 1
    ), "VertexPassthroughLoggingHandler must still fire for VERTEX_AI"


# ---------------------------------------------------------------------------
# 5. End-to-end: success callbacks ACTUALLY fire for the GOOGLE_GENAI path
#
# This is the test PR #24114's reviewer (Greptile P1) flagged as missing.
# The original test mocked `async_success_handler` itself — making the test
# pass even if the callback loop was never reached. This version uses a real
# Logging instance and a real CustomLogger, mocking only the upstream chunk
# parsing so we can drive a fake but valid `result` through.
# ---------------------------------------------------------------------------


def _make_spy_logger():
    """Build a CustomLogger subclass that records whether the success event fired.

    Must be a real CustomLogger because the dispatcher uses
    ``isinstance(callback, CustomLogger)`` to decide whether to route through
    log_success_event / async_log_success_event.
    """
    from litellm.integrations.custom_logger import CustomLogger

    class _SpyCustomLogger(CustomLogger):
        def __init__(self):
            super().__init__()
            self.fired_sync = False
            self.fired_async = False
            self.call_args: List[Any] = []

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.fired_sync = True
            self.call_args.append((kwargs, response_obj, start_time, end_time))

        async def async_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            self.fired_async = True
            self.call_args.append((kwargs, response_obj, start_time, end_time))

    return _SpyCustomLogger()


@pytest.mark.asyncio
async def test_callbacks_actually_fire_for_google_genai_endpoint():
    """Drive the full route → async_success_handler → CustomLogger callback
    chain with the real Logging implementation. Verifies the fix actually
    delivers callbacks (not just that the right code path is taken)."""
    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.pass_through_endpoints.streaming_handler import (
        PassThroughStreamingHandler,
    )
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        EndpointType,
    )

    spy = _make_spy_logger()
    original_callbacks = list(litellm.callbacks)
    original_async = list(litellm._async_success_callback)
    original_sync = list(litellm.success_callback)
    litellm.callbacks = [spy]  # type: ignore[assignment]
    # async_success_handler iterates _async_success_callback, not callbacks,
    # so register on both for belt-and-suspenders.
    litellm._async_success_callback = [spy]  # type: ignore[assignment]
    litellm.success_callback = [spy]  # type: ignore[assignment]
    try:
        real_logging_obj = Logging(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="pass_through_endpoint",
            start_time=datetime.now(),
            litellm_call_id="test-call-id-24097",
            function_id="",
        )
        # Provide a minimal litellm_params so async_success_handler doesn't
        # blow up looking for metadata.
        real_logging_obj.update_environment_variables(
            model="gemini-2.5-pro",
            user="",
            optional_params={},
            litellm_params={"metadata": {}, "api_base": ""},
        )

        # Mock the chunk parser to return a synthetic but valid response so
        # the gemini handler doesn't try to parse real bytes. We're testing
        # callback delivery, not parser correctness.
        from litellm.types.utils import StandardPassThroughResponseObject

        with patch(
            "litellm.proxy.pass_through_endpoints.streaming_handler"
            ".GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks",
            return_value={
                "result": StandardPassThroughResponseObject(response="ok"),
                "kwargs": {},
            },
        ):
            await PassThroughStreamingHandler._route_streaming_logging_to_handler(
                litellm_logging_obj=real_logging_obj,
                passthrough_success_handler_obj=MagicMock(),
                url_route="/models/gemini-2.5-pro:streamGenerateContent",
                request_body={},
                endpoint_type=EndpointType.GOOGLE_GENAI,
                start_time=datetime.now(),
                raw_bytes=[b"data: {}\n"],
                end_time=datetime.now(),
                model="gemini-2.5-pro",
            )

        assert spy.fired_async or spy.fired_sync, (
            "CustomLogger.log_success_event was NEVER called — callbacks are "
            "still silently skipped end-to-end. The fix is incomplete."
        )
    finally:
        litellm.callbacks = original_callbacks
        litellm._async_success_callback = original_async
        litellm.success_callback = original_sync
