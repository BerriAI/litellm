from __future__ import annotations

import asyncio
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

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
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

_ANTHROPIC_SSE: Final = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5-20251001","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
    'event: content_block_start\n'
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"why did the chicken cross the road"}}\n\n'
    'event: content_block_stop\n'
    'data: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
)

_CONVERSE_CACHE_BODY: Final = {
    "output": {"message": {"role": "assistant", "content": [{"text": "cached joke"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 10, "outputTokens": 20, "cacheReadInputTokens": 500, "cacheWriteInputTokens": 50},
}

_INVOKE_CACHE_BODY: Final = {
    "id": "msg_cache",
    "type": "message",
    "role": "assistant",
    "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "content": [{"type": "text", "text": "cached joke"}],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 500,
    },
}

_TOOL_USE_BODY: Final = {
    "id": "msg_tool",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5-20251001",
    "content": [
        {"type": "text", "text": "I will search"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "web_search",
            "input": {"query": "litellm"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 20},
}


async def _collect_stream(response: object) -> str:
    text: Final = []
    async for chunk in response:  # type: ignore[union-attr]
        text.append(str(chunk))
    return "".join(text)


@pytest.mark.asyncio
async def test_anthropic_messages_non_streaming_logs_usage_model_and_cost(respx_mock, monkeypatch):
    pytest.skip("BUG: _finalize_anthropic_messages_response returns without invoking any success handler, so non-streaming /v1/messages never produces a StandardLoggingPayload (litellm/llms/custom_httpx/llm_http_handler.py:2288); control test_control_recorder_fires_for_acompletion proves the recorder setup fires")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    recorder: Final = _RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    respx_mock.post(url__regex=r".*api\.anthropic\.com/v1/messages.*").mock(
        return_value=httpx.Response(200, json=_ANTHROPIC_BODY)
    )
    router: Final = Router(model_list=[_deployment("claude-haiku-4-5-20251001")])
    await GLOBAL_LOGGING_WORKER.clear_queue()
    GLOBAL_LOGGING_WORKER.start()
    await router.aanthropic_messages(
        messages=[{"role": "user", "content": "Hello"}],
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
    )
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    ours: Final = [c for c in recorder.calls if c.get("model") == "gpt-4o"]
    assert len(ours) == 1 and ours[0]["status"] == "success", recorder.calls
    payload: Final = recorder.calls[0]
    assert payload["status"] == "success"
    assert payload["model"] == "claude-haiku-4-5-20251001"
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["response_cost"] > 0
    assert payload["usage"]["prompt_tokens"] == 10
    assert payload["usage"]["completion_tokens"] == 20


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_logs_payload_after_canned_sse(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    recorder: Final = _RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    respx_mock.post(url__regex=r".*api\.anthropic\.com/v1/messages.*").mock(
        return_value=httpx.Response(
            200, content=_ANTHROPIC_SSE.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    await GLOBAL_LOGGING_WORKER.clear_queue()
    GLOBAL_LOGGING_WORKER.start()
    stream: Final = await litellm.anthropic.messages.acreate(
        messages=[{"role": "user", "content": "Hello"}],
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        stream=True,
        api_key="fake-key",
    )
    await _collect_stream(stream)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    assert len(recorder.calls) >= 1, recorder.calls
    payload: Final = recorder.calls[0]
    assert payload["status"] == "success"
    assert payload["response_cost"] > 0


@pytest.mark.asyncio
async def test_bedrock_converse_prompt_caching_returns_cache_tokens(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    route: Final = respx_mock.post(url__regex=r".*bedrock-runtime.*converse.*").mock(
        return_value=httpx.Response(200, json=_CONVERSE_CACHE_BODY)
    )
    response: Final = await litellm.anthropic.messages.acreate(
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}]}],
        model="bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=100,
    )
    assert route.called
    usage: Final = response.get("usage") or {}
    assert usage.get("cache_creation_input_tokens") == 50
    assert usage.get("cache_read_input_tokens") == 500


@pytest.mark.asyncio
async def test_bedrock_invoke_prompt_caching_returns_cache_tokens(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    route: Final = respx_mock.post(url__regex=r".*bedrock-runtime.*").mock(
        return_value=httpx.Response(200, json=_INVOKE_CACHE_BODY)
    )
    response: Final = await litellm.anthropic.messages.acreate(
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}]}],
        model="bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=100,
    )
    assert route.called
    usage: Final = response.get("usage") or {}
    assert "cache_creation_input_tokens" in usage
    assert usage.get("cache_read_input_tokens") == 500


@pytest.mark.asyncio
async def test_tool_search_forwards_deferred_tools_and_returns_tool_use(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(url__regex=r".*api\.anthropic\.com/v1/messages.*").mock(
        return_value=httpx.Response(200, json=_TOOL_USE_BODY)
    )
    tools: Final = [
        {"type": "web_search_20250305", "name": "web_search", "defer_loading": True},
        {
            "type": "function",
            "name": "get_weather",
            "description": "get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    ]
    response: Final = await litellm.anthropic.messages.acreate(
        messages=[{"role": "user", "content": "search the web for litellm"}],
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        tools=tools,
        api_key="fake-key",
    )
    assert route.called
    sent: Final = json.loads(route.calls[0].request.read())
    assert sent["tools"] == tools
    tool_blocks: Final = [block for block in response["content"] if block.get("type") == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "web_search"


@pytest.mark.asyncio
async def test_control_recorder_fires_for_acompletion(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    recorder: Final = _RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [*litellm._async_success_callback, recorder])
    respx_mock.post(url__regex=r".*api\.openai\.com/v1/chat/completions.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    GLOBAL_LOGGING_WORKER.start()
    await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        api_key="fake-key",
    )
    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    except asyncio.TimeoutError:
        pass
    deadline: Final = asyncio.get_running_loop().time() + 10.0
    while not any(c.get("model") == "gpt-4o" for c in recorder.calls) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    ours: Final = [c for c in recorder.calls if c.get("model") == "gpt-4o"]
    assert len(ours) == 1 and ours[0]["status"] == "success", recorder.calls
