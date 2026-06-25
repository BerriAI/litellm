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
