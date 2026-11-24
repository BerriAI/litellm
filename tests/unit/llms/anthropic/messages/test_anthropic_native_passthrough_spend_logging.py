from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import Request, Response

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy._types import UserAPIKeyAuth, hash_token
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import anthropic_proxy_route
from litellm.types.utils import StandardLoggingPayload

_UPSTREAM: Final = "https://api.anthropic.com/v1/messages"
_MODEL: Final = "claude-sonnet-4-5-20250929"
_VIRTUAL_KEY: Final = "sk-native-passthrough"


class _RecordingLogger(CustomLogger):
    def __init__(self, message_id: str) -> None:
        super().__init__()
        self.message_id: Final = message_id
        self.payloads: tuple[StandardLoggingPayload, ...] = ()
        self.received: Final = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        payload: Final = kwargs.get("standard_logging_object")
        if payload is not None and payload["id"] == self.message_id:
            self.payloads = (*self.payloads, payload)
            self.received.set()


def _proxy_request(body: Mapping[str, object]) -> Request:
    request: Final = MagicMock(spec=Request)
    request.method = "POST"
    request.url = httpx.URL("http://proxy/anthropic/v1/messages")
    request.headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    request.scope = {"path": "/anthropic/v1/messages", "type": "http", "method": "POST", "headers": []}
    request.query_params = {}
    request.body = AsyncMock(return_value=json.dumps(body).encode())
    request.json = AsyncMock(return_value=body)
    return request


def _sse(events: tuple[Mapping[str, object], ...]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


async def _wait_for_payload(recorder: _RecordingLogger) -> None:
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.wait_for(recorder.received.wait(), timeout=30.0)


def _assert_spend_payload(
    payload: StandardLoggingPayload, message_id: str, tags: list[str], prompt_tokens: int, completion_tokens: int
) -> None:
    assert payload["id"] == message_id
    assert payload["call_type"] == "pass_through_endpoint"
    assert payload["status"] == "success"
    assert payload["custom_llm_provider"] == "anthropic"
    assert payload["model"] == _MODEL
    assert payload["prompt_tokens"] == prompt_tokens
    assert payload["completion_tokens"] == completion_tokens
    assert payload["total_tokens"] == prompt_tokens + completion_tokens
    assert payload["response_cost"] > 0
    assert payload["request_tags"] == tags
    assert payload["cache_hit"] is not True
    assert payload["startTime"] <= payload["endTime"]
    assert payload["metadata"]["user_api_key_hash"] == hash_token(_VIRTUAL_KEY)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-anthropic-key")
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    logger: Final = _RecordingLogger(f"msg_{uuid.uuid4().hex}")
    monkeypatch.setattr(litellm, "callbacks", [logger])
    monkeypatch.setattr(litellm, "_async_success_callback", [logger])
    return logger


@pytest.mark.asyncio
async def test_native_anthropic_passthrough_logs_usage_tags_and_spend(respx_mock, recorder: _RecordingLogger):
    tags: Final = ["test-tag-1", "test-tag-2"]
    route: Final = respx_mock.post(_UPSTREAM).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": recorder.message_id,
                "type": "message",
                "role": "assistant",
                "model": _MODEL,
                "content": [{"type": "text", "text": "hello test"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )
    )
    response: Final = await anthropic_proxy_route(
        endpoint="v1/messages",
        request=_proxy_request(
            {
                "model": _MODEL,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Say 'hello test' and nothing else"}],
                "litellm_metadata": {"tags": tags},
            }
        ),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=UserAPIKeyAuth(api_key=_VIRTUAL_KEY, token=_VIRTUAL_KEY),
    )
    assert response.status_code == 200
    assert json.loads(response.body)["id"] == recorder.message_id
    outbound: Final = route.calls.last.request
    assert outbound.headers["x-api-key"] == "synthetic-anthropic-key"
    assert json.loads(outbound.content) == {
        "model": _MODEL,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Say 'hello test' and nothing else"}],
    }
    await _wait_for_payload(recorder)
    assert len(recorder.payloads) == 1
    payload: Final = recorder.payloads[0]
    _assert_spend_payload(payload, recorder.message_id, tags, prompt_tokens=11, completion_tokens=7)
    assert payload["api_base"] == _UPSTREAM


@pytest.mark.asyncio
async def test_native_anthropic_passthrough_streaming_logs_usage_tags_and_spend(respx_mock, recorder: _RecordingLogger):
    tags: Final = ["test-tag-stream-1", "test-tag-stream-2"]
    events: Final = (
        {
            "type": "message_start",
            "message": {
                "id": recorder.message_id,
                "type": "message",
                "role": "assistant",
                "model": _MODEL,
                "content": [],
                "usage": {"input_tokens": 11, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello stream test"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    )
    route: Final = respx_mock.post(_UPSTREAM).mock(
        return_value=httpx.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})
    )
    response: Final = await anthropic_proxy_route(
        endpoint="v1/messages",
        request=_proxy_request(
            {
                "model": _MODEL,
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "Say 'hello stream test' and nothing else"}],
                "litellm_metadata": {"tags": tags, "user": "test-user-1"},
            }
        ),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=UserAPIKeyAuth(api_key=_VIRTUAL_KEY, token=_VIRTUAL_KEY),
    )
    assert response.status_code == 200
    streamed: Final = b"".join([chunk async for chunk in response.body_iterator])
    assert b"hello stream test" in streamed
    assert json.loads(route.calls.last.request.content)["stream"] is True
    await _wait_for_payload(recorder)
    assert len(recorder.payloads) == 1
    _assert_spend_payload(recorder.payloads[0], recorder.message_id, tags, prompt_tokens=11, completion_tokens=7)
