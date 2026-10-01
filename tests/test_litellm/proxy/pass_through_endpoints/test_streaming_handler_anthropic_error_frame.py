import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

MODEL = "claude-fable-5"


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _partial_anthropic_stream() -> list[bytes]:
    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_partial",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 29, "output_tokens": 2},
        },
    }
    block_start = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial answer"}}
    return [
        _sse("message_start", message_start),
        _sse("content_block_start", block_start),
        _sse("content_block_delta", delta),
    ]


def _error_frame(error_type: str, message: str) -> bytes:
    return _sse("error", {"type": "error", "error": {"type": error_type, "message": message}})


def _logging_obj() -> MagicMock:
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.model_call_details = {"model": MODEL, "stream": True}
    logging_obj.optional_params = {}
    logging_obj.litellm_params = {}
    logging_obj.litellm_call_id = "test-call-id"
    logging_obj.get_router_model_id.return_value = None
    logging_obj.dispatch_success_handlers = AsyncMock()
    logging_obj.dispatch_failure_handlers = AsyncMock()
    return logging_obj


async def _route(logging_obj: MagicMock, raw_bytes: list[bytes]) -> None:
    await PassThroughStreamingHandler._route_streaming_logging_to_handler(
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=PassThroughEndpointLogging(),
        url_route="/anthropic/v1/messages",
        request_body={"model": MODEL, "stream": True},
        endpoint_type=EndpointType.ANTHROPIC,
        start_time=datetime.now(),
        raw_bytes=raw_bytes,
        end_time=datetime.now(),
        model=MODEL,
    )
    await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type, expected_status",
    [("api_error", 500), ("overloaded_error", 503), ("rate_limit_error", 429)],
)
async def test_in_band_error_frame_is_logged_as_failure_not_success(error_type: str, expected_status: int):
    logging_obj = _logging_obj()

    await _route(logging_obj, [*_partial_anthropic_stream(), _error_frame(error_type, "boom")])

    logging_obj.dispatch_success_handlers.assert_not_awaited()
    logging_obj.dispatch_failure_handlers.assert_awaited_once()
    exception = logging_obj.dispatch_failure_handlers.await_args.args[0]
    assert exception.status_code == expected_status
    assert "boom" in str(exception)
    logging_obj.record_partial_usage_for_failure.assert_called_once()


@pytest.mark.asyncio
async def test_stream_without_error_frame_still_logs_success():
    logging_obj = _logging_obj()

    await _route(logging_obj, _partial_anthropic_stream())

    logging_obj.dispatch_success_handlers.assert_awaited_once()
    logging_obj.dispatch_failure_handlers.assert_not_awaited()
