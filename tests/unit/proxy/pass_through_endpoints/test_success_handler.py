import asyncio
import json
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


# Helper function to mock async iteration
async def aiter_mock(iterable):
    for item in iterable:
        yield item


@pytest.mark.usefixtures("_drain_logging_worker", "_vcr_outcome_gate")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_type,url_route",
    [
        (
            EndpointType.VERTEX_AI,
            "v1/projects/pathrise-convert-1606954137718/locations/us-central1/publishers/google/models/gemini-1.0-pro:generateContent",
        ),
        (EndpointType.ANTHROPIC, "/v1/messages"),
    ],
)
async def test_chunk_processor_yields_raw_bytes(endpoint_type, url_route):
    """
    Test that the chunk_processor yields raw bytes

    This is CRITICAL for pass throughs streaming with Vertex AI and Anthropic
    """
    # Mock inputs
    response = AsyncMock(spec=httpx.Response)
    response.status_code = 200
    raw_chunks = [
        b'{"id": "1", "content": "Hello"}',
        b'{"id": "2", "content": "World"}',
        b'\n\ndata: {"id": "3"}',  # Testing different byte formats
    ]

    # Mock aiter_bytes to return an async generator
    async def mock_aiter_bytes():
        for chunk in raw_chunks:
            yield chunk

    response.aiter_bytes = mock_aiter_bytes

    request_body = {"key": "value"}
    litellm_logging_obj = MagicMock()
    start_time = datetime.now()
    passthrough_success_handler_obj = MagicMock()
    litellm_logging_obj.async_success_handler = AsyncMock()

    # Capture yielded chunks and perform detailed assertions
    received_chunks = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body=request_body,
        litellm_logging_obj=litellm_logging_obj,
        endpoint_type=endpoint_type,
        start_time=start_time,
        passthrough_success_handler_obj=passthrough_success_handler_obj,
        url_route=url_route,
    ):
        # Assert each chunk is bytes
        assert isinstance(chunk, bytes), f"Chunk should be bytes, got {type(chunk)}"
        # Assert no decoding/encoding occurred (chunk should be exactly as input)
        assert chunk in raw_chunks, (
            f"Chunk {chunk} was modified during processing. For pass throughs streaming, chunks should be raw bytes"
        )
        received_chunks.append(chunk)

    # Assert all chunks were processed
    assert len(received_chunks) == len(raw_chunks), "Not all chunks were processed"

    # collected chunks all together
    assert b"".join(received_chunks) == b"".join(raw_chunks), "Collected chunks do not match raw chunks"


@pytest.mark.usefixtures("_drain_logging_worker", "_vcr_outcome_gate")
@pytest.mark.asyncio
async def test_route_streaming_logging_runs_async_handler_for_sdk_passthrough():
    """
    SDK pass-through streaming (anthropic_messages, google generate_content) must run
    the async success handler so async-only loggers record the assembled stream.

    Regression for duplicate-trace dedupe: dispatch_success_handlers treated these as
    sync SDK requests because call_type is not ``pass_through_endpoint`` and
    litellm_params carries no ``acompletion`` flag, so only the sync success_handler
    ran and CustomLogger.async_log_success_event never fired.
    """
    import time

    from litellm.types.utils import CallTypes

    logging_obj = LiteLLMLoggingObj(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type=CallTypes.anthropic_messages.value,
        start_time=time.time(),
        litellm_call_id="test-id",
        function_id="fn",
    )
    logging_obj.model_call_details["litellm_params"] = {"anthropic_messages": True}

    with (
        patch.object(
            PassThroughStreamingHandler,
            "_build_passthrough_logging_result",
            return_value=({"id": "slp"}, {}),
        ),
        patch.object(logging_obj, "async_success_handler", new_callable=AsyncMock) as mock_async,
        patch.object(logging_obj, "success_handler", new_callable=MagicMock) as mock_sync,
        patch.object(
            logging_obj,
            "_should_run_sync_callbacks_for_async_calls",
            return_value=False,
        ),
    ):
        await PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/v1/messages",
            request_body={},
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=datetime.now(),
            raw_bytes=[],
            end_time=datetime.now(),
        )

    mock_async.assert_awaited_once()
    mock_sync.assert_not_called()


@pytest.mark.usefixtures("_drain_logging_worker", "_vcr_outcome_gate")
@pytest.mark.asyncio
async def test_handle_logging_runs_async_handler_for_passthrough():
    """
    Non-streaming pass-through logging (_handle_logging) must always run the
    async success handler so async-only loggers (e.g. the proxy spend logger)
    record the request.

    _handle_logging is only ever reached from pass_through_async_success_handler
    (an async context), so it forces async dispatch via prefer_async_handlers.
    This pins that contract independent of the call-type classification: even a
    call_type that _is_sync_litellm_request would classify as sync (here
    "completion" with no async marker in litellm_params) must still reach
    async_success_handler. Without prefer_async_handlers=True the sync-only
    branch would return early and async_log_success_event would never fire.
    """
    import time

    from litellm.types.utils import CallTypes

    logging_obj = LiteLLMLoggingObj(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type=CallTypes.completion.value,
        start_time=time.time(),
        litellm_call_id="test-id",
        function_id="fn",
    )
    logging_obj.model_call_details["litellm_params"] = {}

    handler = PassThroughEndpointLogging()

    with (
        patch.object(logging_obj, "async_success_handler", new_callable=AsyncMock) as mock_async,
        patch.object(logging_obj, "success_handler", new_callable=MagicMock) as mock_sync,
        patch.object(
            logging_obj,
            "_should_run_sync_callbacks_for_async_calls",
            return_value=False,
        ),
    ):
        await handler._handle_logging(
            logging_obj=logging_obj,
            standard_logging_response_object={"id": "slp"},
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
        )

    mock_async.assert_awaited_once()
    mock_sync.assert_not_called()


@pytest.mark.usefixtures("_drain_logging_worker", "_vcr_outcome_gate")
def test_convert_raw_bytes_to_str_lines():
    """
    Test that the _convert_raw_bytes_to_str_lines method correctly converts raw bytes to a list of strings
    """
    # Test case 1: Single chunk
    raw_bytes = [b'data: {"content": "Hello"}\n']
    result = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)
    assert result == ['data: {"content": "Hello"}']

    # Test case 2: Multiple chunks
    raw_bytes = [b'data: {"content": "Hello"}\n', b'data: {"content": "World"}\n']
    result = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)
    assert result == ['data: {"content": "Hello"}', 'data: {"content": "World"}']

    # Test case 3: Empty input
    raw_bytes = []
    result = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)
    assert result == []

    # Test case 4: Chunks with empty lines
    raw_bytes = [b'data: {"content": "Hello"}\n\n', b'\ndata: {"content": "World"}\n']
    result = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)
    assert result == ['data: {"content": "Hello"}', 'data: {"content": "World"}']


@pytest.fixture()
async def _drain_logging_worker():
    """
    The logging queue is bound to the running loop, so anything left queued when a test's loop
    goes away is carried onto the next loop and fires against that test's callbacks.
    """
    GLOBAL_LOGGING_WORKER.start()
    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    except asyncio.TimeoutError:
        pass
    await GLOBAL_LOGGING_WORKER.stop()
    yield


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.mark.usefixtures("_drain_logging_worker", "_vcr_outcome_gate")
@pytest.mark.asyncio
async def test_vertex_ai_anthropic_streaming_cost_injection_enabled():
    """
    Test that cost is injected into Vertex AI streamRawPredict streaming chunks
    when include_cost_in_streaming_usage is enabled.
    """
    # Enable cost injection
    original_value = getattr(litellm, "include_cost_in_streaming_usage", False)
    litellm.include_cost_in_streaming_usage = True

    try:
        # Mock response with Anthropic SSE format chunks
        response = AsyncMock(spec=httpx.Response)
        response.status_code = 200

        # Create chunks with message_delta event containing usage
        chunks_with_usage = [
            b'data: {"type": "content_block_delta", "delta": {"text": "Hello"}}\n\n',
            b'data: {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 5}}\n\n',
            b'data: {"type": "content_block_delta", "delta": {"text": " world"}}\n\n',
        ]

        async def mock_aiter_bytes():
            for chunk in chunks_with_usage:
                yield chunk

        response.aiter_bytes = mock_aiter_bytes

        # Setup logging object with model info
        litellm_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        litellm_logging_obj.litellm_params = {}
        litellm_logging_obj.model_call_details = {"model": "claude-sonnet-4@20250514"}
        litellm_logging_obj.completion_start_time = None
        litellm_logging_obj.async_success_handler = AsyncMock()

        request_body = {"model": "claude-sonnet-4@20250514"}
        start_time = datetime.now()
        passthrough_success_handler_obj = MagicMock(spec=PassThroughEndpointLogging)

        url_route = "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict"

        # Mock completion_cost to return a test cost value
        with patch("litellm.completion_cost", return_value=0.00015):
            received_chunks = []
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=response,
                request_body=request_body,
                litellm_logging_obj=litellm_logging_obj,
                endpoint_type=EndpointType.VERTEX_AI,
                start_time=start_time,
                passthrough_success_handler_obj=passthrough_success_handler_obj,
                url_route=url_route,
            ):
                received_chunks.append(chunk)

        # Verify that cost was injected into the message_delta chunk
        cost_injected = False
        for chunk in received_chunks:
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8", errors="ignore")
                if "message_delta" in chunk_str and "cost" in chunk_str:
                    # Parse the chunk to verify cost was added
                    for line in chunk_str.split("\n"):
                        if line.startswith("data:") and "message_delta" in line:
                            json_part = line.split("data:", 1)[1].strip()
                            if json_part and json_part != "[DONE]":
                                try:
                                    obj = json.loads(json_part)
                                    if obj.get("type") == "message_delta" and "usage" in obj and "cost" in obj["usage"]:
                                        assert obj["usage"]["cost"] == 0.00015
                                        cost_injected = True
                                except json.JSONDecodeError:
                                    pass

        assert cost_injected, "Cost was not injected into message_delta chunk"

    finally:
        # Restore original value
        litellm.include_cost_in_streaming_usage = original_value


@pytest.mark.usefixtures("_drain_logging_worker", "_vcr_outcome_gate")
@pytest.mark.asyncio
async def test_vertex_ai_anthropic_streaming_cost_injection_disabled():
    """
    Test that cost is NOT injected when include_cost_in_streaming_usage is disabled.
    """
    # Disable cost injection
    original_value = getattr(litellm, "include_cost_in_streaming_usage", False)
    litellm.include_cost_in_streaming_usage = False

    try:
        # Mock response with Anthropic SSE format chunks
        response = AsyncMock(spec=httpx.Response)
        response.status_code = 200

        chunks_with_usage = [
            b'data: {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 5}}\n\n',
        ]

        async def mock_aiter_bytes():
            for chunk in chunks_with_usage:
                yield chunk

        response.aiter_bytes = mock_aiter_bytes

        litellm_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        litellm_logging_obj.litellm_params = {}
        litellm_logging_obj.model_call_details = {"model": "claude-sonnet-4@20250514"}
        litellm_logging_obj.completion_start_time = None
        litellm_logging_obj.async_success_handler = AsyncMock()

        request_body = {"model": "claude-sonnet-4@20250514"}
        start_time = datetime.now()
        passthrough_success_handler_obj = MagicMock(spec=PassThroughEndpointLogging)

        url_route = "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict"

        received_chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
            endpoint_type=EndpointType.VERTEX_AI,
            start_time=start_time,
            passthrough_success_handler_obj=passthrough_success_handler_obj,
            url_route=url_route,
        ):
            received_chunks.append(chunk)

        # Verify that cost was NOT injected
        cost_found = False
        for chunk in received_chunks:
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8", errors="ignore")
                if "cost" in chunk_str:
                    cost_found = True

        assert not cost_found, "Cost should not be injected when feature is disabled"

    finally:
        # Restore original value
        litellm.include_cost_in_streaming_usage = original_value


@pytest.mark.usefixtures("_drain_logging_worker", "_vcr_outcome_gate")
@pytest.mark.asyncio
async def test_vertex_ai_anthropic_streaming_cost_injection_no_usage_chunk():
    """
    Test that chunks without usage are not modified.
    """
    original_value = getattr(litellm, "include_cost_in_streaming_usage", False)
    litellm.include_cost_in_streaming_usage = True

    try:
        response = AsyncMock(spec=httpx.Response)
        response.status_code = 200

        # Chunks without usage (should not be modified)
        chunks_without_usage = [
            b'data: {"type": "content_block_delta", "delta": {"text": "Hello"}}\n\n',
            b'data: {"type": "content_block_start", "index": 0}\n\n',
        ]

        async def mock_aiter_bytes():
            for chunk in chunks_without_usage:
                yield chunk

        response.aiter_bytes = mock_aiter_bytes

        litellm_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        litellm_logging_obj.litellm_params = {}
        litellm_logging_obj.model_call_details = {"model": "claude-sonnet-4@20250514"}
        litellm_logging_obj.completion_start_time = None
        litellm_logging_obj.async_success_handler = AsyncMock()

        request_body = {"model": "claude-sonnet-4@20250514"}
        start_time = datetime.now()
        passthrough_success_handler_obj = MagicMock(spec=PassThroughEndpointLogging)

        url_route = "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict"

        received_chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
            endpoint_type=EndpointType.VERTEX_AI,
            start_time=start_time,
            passthrough_success_handler_obj=passthrough_success_handler_obj,
            url_route=url_route,
        ):
            received_chunks.append(chunk)

        # Verify chunks remain unchanged (no cost injection attempted)
        assert len(received_chunks) == len(chunks_without_usage)
        # Chunks should be exactly as input since they don't contain usage
        for i, chunk in enumerate(received_chunks):
            assert chunk == chunks_without_usage[i]

    finally:
        litellm.include_cost_in_streaming_usage = original_value


@pytest.mark.usefixtures("_drain_logging_worker", "_vcr_outcome_gate")
@pytest.mark.asyncio
async def test_vertex_ai_anthropic_streaming_model_extraction():
    """
    Test that model name is correctly extracted for cost calculation.
    """
    original_value = getattr(litellm, "include_cost_in_streaming_usage", False)
    litellm.include_cost_in_streaming_usage = True

    try:
        response = AsyncMock(spec=httpx.Response)
        response.status_code = 200

        chunks = [
            b'data: {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 5}}\n\n',
        ]

        async def mock_aiter_bytes():
            for chunk in chunks:
                yield chunk

        response.aiter_bytes = mock_aiter_bytes

        litellm_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        litellm_logging_obj.litellm_params = {}
        litellm_logging_obj.model_call_details = {}
        litellm_logging_obj.completion_start_time = None
        litellm_logging_obj.async_success_handler = AsyncMock()

        # Test model extraction from request body
        request_body = {"model": "claude-sonnet-4@20250514"}
        start_time = datetime.now()
        passthrough_success_handler_obj = MagicMock(spec=PassThroughEndpointLogging)

        url_route = "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict"

        with patch("litellm.completion_cost") as mock_cost:
            mock_cost.return_value = 0.0001
            received_chunks = []
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=response,
                request_body=request_body,
                litellm_logging_obj=litellm_logging_obj,
                endpoint_type=EndpointType.VERTEX_AI,
                start_time=start_time,
                passthrough_success_handler_obj=passthrough_success_handler_obj,
                url_route=url_route,
            ):
                received_chunks.append(chunk)

            # Verify completion_cost was called with the correct model
            assert mock_cost.called
            call_args = mock_cost.call_args
            assert call_args[1]["model"] == "claude-sonnet-4@20250514"

    finally:
        litellm.include_cost_in_streaming_usage = original_value


_LIVE_ROUTE: Final = "/vertex_ai/live"
_LIVE_MODEL: Final = "gemini-live-2.5-flash"
_START: Final = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_END: Final = datetime(2025, 1, 2, 3, 4, 9, tzinfo=timezone.utc)
_FIRST_TURN_USAGE: Final = {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14}
_SECOND_TURN_USAGE: Final = {"promptTokenCount": 30, "candidatesTokenCount": 6, "totalTokenCount": 36}


def _logging_obj() -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "WebSocket connection"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=_START,
        litellm_call_id="call-live",
        function_id="websocket_passthrough",
    )


def _normalize_live_session(response_body: dict | list[dict[str, object]] | None) -> dict:
    return PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
        httpx_response=httpx.Response(200, request=httpx.Request("GET", f"https://proxy.example.test{_LIVE_ROUTE}")),
        response_body=response_body,
        request_body={},
        logging_obj=_logging_obj(),
        url_route=_LIVE_ROUTE,
        result="websocket_connection_successful",
        start_time=_START,
        end_time=_END,
        cache_hit=False,
        model=_LIVE_MODEL,
    )


@pytest.mark.usefixtures("local_model_cost_map")
def test_vertex_ai_live_route_sums_usage_of_every_turn_in_the_websocket_frames() -> None:
    normalized = _normalize_live_session(
        [
            {"setupComplete": {}},
            {"serverContent": {"modelTurn": {"parts": [{"text": "hello"}]}}},
            {"usageMetadata": _FIRST_TURN_USAGE},
            {"serverContent": {"modelTurn": {"parts": [{"text": "goodbye"}]}}},
            {"usageMetadata": _SECOND_TURN_USAGE},
        ]
    )

    response = normalized["standard_logging_response_object"]
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (40, 10, 50)
    assert response.model == _LIVE_MODEL
    assert normalized["kwargs"] == {"model": _LIVE_MODEL, "custom_llm_provider": "vertex_ai"}


@pytest.mark.parametrize(
    "response_body",
    [
        pytest.param({"usageMetadata": _FIRST_TURN_USAGE}, id="single-json-object"),
        pytest.param(None, id="no-body"),
        pytest.param([{"setupComplete": {}}], id="frames-without-usage"),
    ],
)
@pytest.mark.usefixtures("local_model_cost_map")
def test_vertex_ai_live_route_without_usage_frames_yields_no_logging_response(
    response_body: dict | list[dict[str, object]] | None,
) -> None:
    normalized = _normalize_live_session(response_body)

    assert normalized["standard_logging_response_object"] is None
    assert normalized["kwargs"] == {"model": _LIVE_MODEL}
