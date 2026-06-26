"""Regression tests: pass-through streaming must stamp completion_start_time on the
first chunk so the time-to-first-token metric reflects TTFT.

Without it completion_start_time stays None and litellm_logging falls back to
end_time, making litellm_llm_api_time_to_first_token_metric equal full generation
latency for every /v1/messages (and other pass-through) streaming request."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def _make_streaming_response(chunks):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200

    async def _aiter_bytes():
        for c in chunks:
            yield c

    mock.aiter_bytes = _aiter_bytes
    return mock


def _logging_obj():
    obj = MagicMock()
    obj.completion_start_time = None
    obj.model_call_details = {}
    return obj


@pytest.mark.asyncio
async def test_completion_start_time_stamped_on_first_chunk():
    response = _make_streaming_response([b"event-1", b"event-2", b"event-3"])
    logging_obj = _logging_obj()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ):
        gen = PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(),
            url_route="/v1/messages",
        )

        first = await gen.__anext__()
        # the whole point: TTFT must be stamped on the first chunk, not at end_time
        assert first == b"event-1"
        assert isinstance(logging_obj.completion_start_time, datetime)
        stamped = logging_obj.completion_start_time
        assert logging_obj.model_call_details["completion_start_time"] == stamped

        async for _ in gen:
            pass
        await asyncio.sleep(0)

    # later chunks must not move it forward (TTFT is first-token, not last)
    assert logging_obj.completion_start_time == stamped


@pytest.mark.asyncio
async def test_completion_start_time_stamped_on_cost_injection_path():
    response = _make_streaming_response([b'data: {"x": 1}\n\n'])
    logging_obj = _logging_obj()

    with (
        patch.object(
            PassThroughStreamingHandler,
            "_route_streaming_logging_to_handler",
            new=AsyncMock(),
        ),
        patch.object(litellm, "include_cost_in_streaming_usage", True),
        patch.object(
            PassThroughStreamingHandler,
            "_extract_model_for_cost_injection",
            return_value="claude-sonnet-4-6",
        ),
    ):
        received = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-sonnet-4-6"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(),
            url_route="/v1/messages",
        ):
            received.append(chunk)
        await asyncio.sleep(0)

    assert received  # cost-injection branch was exercised
    assert isinstance(logging_obj.completion_start_time, datetime)
