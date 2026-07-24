"""Tests for litellm/a2a_protocol/main.py non-streaming send behavior."""

import pytest

pytest.importorskip("a2a.compat.v0_3.conversions")

from a2a.compat.v0_3 import conversions as _conv
from a2a.compat.v0_3.types import MessageSendParams, SendMessageRequest

from litellm.a2a_protocol.main import _send_message


def _request() -> SendMessageRequest:
    params = MessageSendParams(
        message={
            "messageId": "m1",
            "role": "user",
            "parts": [{"kind": "text", "text": "hi"}],
        }
    )
    return SendMessageRequest(id="r1", params=params)


def _message_stream_response():
    sr = _conv.pb2_v10.StreamResponse()
    sr.message.message_id = "reply-1"
    sr.message.role = _conv.pb2_v10.Role.ROLE_AGENT
    sr.message.parts.add().text = "hello back"
    return sr


def _status_update_stream_response():
    sr = _conv.pb2_v10.StreamResponse()
    sr.status_update.task_id = "t1"
    sr.status_update.context_id = "c1"
    return sr


class _FakeClient:
    def __init__(self, *events):
        self._events = events

    async def send_message(self, _pb_request):
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_send_message_returns_message_result():
    response = await _send_message(_FakeClient(_message_stream_response()), _request())
    result = response.root.result
    assert type(result).__name__ == "Message"
    assert response.root.id == "r1"


@pytest.mark.asyncio
async def test_send_message_rejects_update_event_final_with_runtime_error():
    with pytest.raises(RuntimeError, match="Message or Task"):
        await _send_message(_FakeClient(_status_update_stream_response()), _request())


@pytest.mark.asyncio
async def test_streaming_trace_id_prefers_logging_trace_id():
    """The streaming X-LiteLLM-Trace-Id must use the logging object's trace id (same
    as the non-streaming path), not the JSON-RPC request id, so traces correlate."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from a2a.compat.v0_3.types import (
        MessageSendParams,
        SendStreamingMessageRequest,
    )

    from litellm.a2a_protocol import main as a2a_main
    from litellm.litellm_core_utils.litellm_logging import Logging

    request = SendStreamingMessageRequest(
        id="rpc-1",
        params=MessageSendParams(
            message={
                "messageId": "m1",
                "role": "user",
                "parts": [{"kind": "text", "text": "hi"}],
            }
        ),
    )
    logging_obj = MagicMock(spec=Logging)
    logging_obj.litellm_trace_id = "trace-from-logging"

    captured: dict = {}

    async def _capture(*, base_url, extra_headers=None, streaming=False, **_):
        captured["extra_headers"] = extra_headers
        raise RuntimeError("stop")

    with patch.object(
        a2a_main, "create_a2a_client", new=AsyncMock(side_effect=_capture)
    ):
        with pytest.raises(RuntimeError, match="stop"):
            async for _ in a2a_main.asend_message_streaming(
                request=request,
                api_base="http://upstream.local",
                litellm_logging_obj=logging_obj,
            ):
                pass

    assert captured["extra_headers"]["X-LiteLLM-Trace-Id"] == "trace-from-logging"
