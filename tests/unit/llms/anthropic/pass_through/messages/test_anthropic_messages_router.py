from __future__ import annotations

import json
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router import Router
from litellm.utils import CustomLogger

_ANTHROPIC_BODY: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5-20251001",
    "content": [{"type": "text", "text": "why did the chicken cross the road"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 20},
}

_CONVERSE_BODY: Final = {
    "output": {"message": {"role": "assistant", "content": [{"text": "why did the chicken cross the road"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 10, "outputTokens": 20},
}

_OPENAI_BODY: Final = {
    "id": "resp_1",
    "status": "completed",
    "created_at": 1700000000,
    "model": "gpt-4.1-mini",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "why did the chicken cross the road"}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
}


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.calls: Final[list[dict]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.calls.append(kwargs.get("standard_logging_object") or {})


def _deployment(model: str, **params: object) -> dict:
    return {"model_name": model, "litellm_params": {"model": model, "api_key": "fake-key", **params}}


@pytest.mark.asyncio
async def test_router_aanthropic_messages_non_streaming_posts_anthropic_body(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(url__regex=r".*api\.anthropic\.com/v1/messages.*").mock(
        return_value=httpx.Response(200, json=_ANTHROPIC_BODY)
    )
    router: Final = Router(model_list=[_deployment("claude-haiku-4-5-20251001")])
    response: Final = await router.aanthropic_messages(
        messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
    )
    assert route.called
    sent: Final = json.loads(route.calls[0].request.read())
    assert sent["max_tokens"] == 100
    assert sent["messages"] == [{"role": "user", "content": "Hello, can you tell me a short joke?"}]
    assert response["id"] == "msg_1"
    assert response["role"] == "assistant"
    assert response["content"][0]["text"] == "why did the chicken cross the road"
    assert response["model"] == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_router_aanthropic_messages_latency_routing_forwards_user_id(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(url__regex=r".*api\.anthropic\.com/v1/messages.*").mock(
        return_value=httpx.Response(200, json=_ANTHROPIC_BODY)
    )
    router: Final = Router(
        model_list=[_deployment("claude-haiku-4-5-20251001")],
        routing_strategy="latency-based-routing",
    )
    response: Final = await router.aanthropic_messages(
        messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        metadata={"user_id": "hello"},
    )
    assert route.called
    sent: Final = json.loads(route.calls[0].request.read())
    assert sent["metadata"] == {"user_id": "hello"}
    assert response["role"] == "assistant"


@pytest.mark.asyncio
async def test_router_aanthropic_messages_falls_back_after_first_deployment_500(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    first: Final = respx_mock.post(url__regex=r".*api\.anthropic\.com/v1/messages.*").mock(
        return_value=httpx.Response(
            500, json={"error": {"type": "api_error", "message": "boom"}}
        )
    )
    second: Final = respx_mock.post(url__regex=r".*bedrock-runtime.*").mock(
        return_value=httpx.Response(200, json=_CONVERSE_BODY)
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    router: Final = Router(
        model_list=[
            _deployment("anthropic/claude-opus-4-7"),
            {
                "model_name": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
            },
        ],
        fallbacks=[{"anthropic/claude-opus-4-7": ["bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"]}],
    )
    response: Final = await router.aanthropic_messages(
        messages=[{"role": "user", "content": "Hello"}],
        model="anthropic/claude-opus-4-7",
        max_tokens=100,
    )
    assert first.called
    assert second.called
    assert response["output"]["message"]["content"][0]["text"] == "why did the chicken cross the road"


@pytest.mark.asyncio
async def test_router_aanthropic_messages_bedrock_converse_posts_signed_request(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    route: Final = respx_mock.post(url__regex=r".*bedrock-runtime.*").mock(
        return_value=httpx.Response(200, json=_CONVERSE_BODY)
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "litellm_params": {"model": "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
            }
        ]
    )
    response: Final = await router.aanthropic_messages(
        messages=[{"role": "user", "content": "Hello"}],
        model="bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=100,
    )
    assert route.called
    sent_request: Final = route.calls[0].request
    assert "authorization" in sent_request.headers
    sent: Final = json.loads(sent_request.read())
    assert sent["messages"] == [{"role": "user", "content": [{"text": "Hello"}]}]
    assert response["role"] == "assistant"


def test_sync_openai_bridge_anthropic_messages_returns_content_blocks(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(url__regex=r".*api\.openai\.com/v1/responses.*").mock(
        return_value=httpx.Response(200, json=_OPENAI_BODY)
    )
    response: Final = litellm.anthropic.messages.create(
        messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
        model="openai/gpt-4.1-mini",
        max_tokens=100,
        api_key="fake-key",
    )
    assert route.called
    sent: Final = json.loads(route.calls[0].request.read())
    assert sent["model"] == "gpt-4.1-mini"
    assert isinstance(response, dict)
    assert response["content"][0]["text"] == "why did the chicken cross the road"
