from __future__ import annotations

import asyncio
import base64
import json
import struct
import uuid
from collections.abc import AsyncIterable, Mapping
from typing import Final
from zlib import crc32

import httpx
import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router
from litellm.types.llms.anthropic import (
    AnthropicMessagesTextParam,
    AnthropicMessagesTool,
    AnthropicMessagesUserMessageParam,
    AnthropicToolSearchToolRegex,
)
from litellm.types.utils import StandardLoggingPayload

_ALIAS: Final = "claude-special-alias"
_ANTHROPIC_MODEL: Final = "claude-haiku-4-5-20251001"
_BEDROCK_MODEL: Final = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
_BEDROCK_SONNET: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_OPENAI_MODEL: Final = "openai/gpt-4.1-mini"
_JOKE: Final = "why did the chicken cross the road"
_PROMPT: Final = "Hello, can you tell me a short joke?"
_ANTHROPIC_URL: Final = r".*api\.anthropic\.com/v1/messages.*"
_OPENAI_RESPONSES_URL: Final = r".*api\.openai\.com/v1/responses.*"
_BEDROCK_INVOKE_URL: Final = r".*bedrock-runtime.*/invoke$"
_BEDROCK_INVOKE_STREAM_URL: Final = r".*bedrock-runtime.*/invoke-with-response-stream$"
_BEDROCK_CONVERSE_URL: Final = r".*bedrock-runtime.*/converse$"
_BEDROCK_CONVERSE_STREAM_URL: Final = r".*bedrock-runtime.*/converse-stream$"


def _anthropic_body(model: str = _ANTHROPIC_MODEL, msg_id: str = "msg_1") -> Mapping[str, object]:
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": _JOKE}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


_CONVERSE_BODY: Final = {
    "output": {"message": {"role": "assistant", "content": [{"text": _JOKE}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
}

_OPENAI_BODY: Final = {
    "id": "resp_1",
    "object": "response",
    "status": "completed",
    "created_at": 1700000000,
    "model": "gpt-4.1-mini",
    "output": [
        {
            "type": "message",
            "id": "msg_out_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": _JOKE, "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
}

_STREAM_EVENTS: Final = (
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": _ANTHROPIC_MODEL,
            "content": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _JOKE}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 20}},
    {"type": "message_stop"},
)


class _RecordingLogger(CustomLogger):
    def __init__(self, messages: list[AnthropicMessagesUserMessageParam]) -> None:
        super().__init__()
        self.messages: Final = messages
        self.payloads: tuple[StandardLoggingPayload, ...] = ()
        self.received: Final = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        payload: Final = kwargs.get("standard_logging_object")
        if payload is not None and payload["messages"] == self.messages:
            self.payloads = (*self.payloads, payload)
            self.received.set()


def _unique_messages() -> list[AnthropicMessagesUserMessageParam]:
    return [{"role": "user", "content": f"{_PROMPT} {uuid.uuid4().hex}"}]


def _router(model_name: str, model: str, **router_kwargs: object) -> Router:
    return Router(
        model_list=[{"model_name": model_name, "litellm_params": {"model": model, "api_key": "fake-key"}}],
        **router_kwargs,
    )


def _event_frame(event_type: str, payload: Mapping[str, object]) -> bytes:
    def header(name: str, value: str) -> bytes:
        name_b: Final = name.encode()
        value_b: Final = value.encode()
        return (
            struct.pack("!B", len(name_b)) + name_b + struct.pack("!B", 7) + struct.pack("!H", len(value_b)) + value_b
        )

    payload_b: Final = json.dumps(payload).encode()
    headers_b: Final = (
        header(":event-type", event_type)
        + header(":content-type", "application/json")
        + header(":message-type", "event")
    )
    prelude: Final = struct.pack("!II", 16 + len(headers_b) + len(payload_b), len(headers_b))
    prelude_crc: Final = crc32(prelude) & 0xFFFFFFFF
    message: Final = struct.pack("!I", prelude_crc) + headers_b + payload_b
    return prelude + message + struct.pack("!I", crc32(message, prelude_crc) & 0xFFFFFFFF)


def _invoke_stream_body() -> bytes:
    return b"".join(
        _event_frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()})
        for event in _STREAM_EVENTS
    )


def _anthropic_sse_body() -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in _STREAM_EVENTS).encode()


def _converse_stream_body() -> bytes:
    return (
        _event_frame("messageStart", {"role": "assistant"})
        + _event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": _JOKE}})
        + _event_frame("contentBlockStop", {"contentBlockIndex": 0})
        + _event_frame("messageStop", {"stopReason": "end_turn"})
        + _event_frame(
            "metadata",
            {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 20,
                    "totalTokens": 530,
                    "cacheReadInputTokens": 500,
                    "cacheWriteInputTokens": 0,
                },
                "metrics": {"latencyMs": 10},
            },
        )
    )


def _set_fake_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")


def _sse_events(raw: str) -> tuple[Mapping[str, object], ...]:
    return tuple(json.loads(line[len("data: ") :]) for line in raw.splitlines() if line.startswith("data: "))


async def _stream_events(stream: object) -> tuple[Mapping[str, object], ...]:
    assert isinstance(stream, AsyncIterable), type(stream)
    chunks: Final = [chunk async for chunk in stream]
    raw: Final = "".join(chunk.decode() for chunk in chunks if isinstance(chunk, bytes))
    dict_events: Final = tuple(chunk for chunk in chunks if isinstance(chunk, Mapping))
    return _sse_events(raw) + dict_events


async def _wait_for_payload(recorder: _RecordingLogger) -> None:
    await asyncio.wait_for(recorder.received.wait(), timeout=30.0)


def _assert_anthropic_message(response: object, model: str) -> None:
    assert isinstance(response, dict), type(response)
    assert response["type"] == "message"
    assert response["role"] == "assistant"
    assert response["model"] == model
    assert isinstance(response["id"], str) and response["id"]
    block: Final = response["content"][0]
    assert isinstance(block, dict), type(block)
    assert block["type"] == "text"
    assert block["text"] == _JOKE


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@pytest.mark.asyncio
async def test_router_aanthropic_messages_non_streaming_posts_anthropic_body(respx_mock):
    route: Final = respx_mock.post(url__regex=_ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_body())
    )
    response: Final = await _router(_ALIAS, _ANTHROPIC_MODEL).aanthropic_messages(
        messages=[{"role": "user", "content": _PROMPT}],
        model=_ALIAS,
        max_tokens=100,
    )
    sent: Final = json.loads(route.calls.last.request.read())
    assert sent["model"] == _ANTHROPIC_MODEL
    assert sent["max_tokens"] == 100
    assert sent["messages"] == [{"role": "user", "content": _PROMPT}]
    _assert_anthropic_message(response, _ANTHROPIC_MODEL)


@pytest.mark.asyncio
async def test_router_aanthropic_messages_latency_routing_forwards_user_id(respx_mock):
    route: Final = respx_mock.post(url__regex=_ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_body())
    )
    response: Final = await _router(
        _ALIAS, _ANTHROPIC_MODEL, routing_strategy="latency-based-routing"
    ).aanthropic_messages(
        messages=[{"role": "user", "content": _PROMPT}],
        model=_ALIAS,
        max_tokens=100,
        metadata={"user_id": "hello"},
    )
    sent: Final = json.loads(route.calls.last.request.read())
    assert sent["model"] == _ANTHROPIC_MODEL
    assert sent["metadata"] == {"user_id": "hello"}
    _assert_anthropic_message(response, _ANTHROPIC_MODEL)


@pytest.mark.asyncio
async def test_router_aanthropic_messages_falls_back_to_bedrock_after_anthropic_401(respx_mock, monkeypatch):
    _set_fake_aws_env(monkeypatch)
    anthropic_route: Final = respx_mock.post(url__regex=_ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        )
    )
    bedrock_route: Final = respx_mock.post(url__regex=_BEDROCK_INVOKE_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_body(model=_BEDROCK_SONNET, msg_id="msg_bedrock"))
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "anthropic/claude-opus-4-7",
                "litellm_params": {"model": "anthropic/claude-opus-4-7", "api_key": "bad-key"},
            },
            {"model_name": f"bedrock/{_BEDROCK_SONNET}", "litellm_params": {"model": f"bedrock/{_BEDROCK_SONNET}"}},
        ],
        fallbacks=[{"anthropic/claude-opus-4-7": [f"bedrock/{_BEDROCK_SONNET}"]}],
    )
    response: Final = await router.aanthropic_messages(
        messages=[{"role": "user", "content": _PROMPT}],
        model="anthropic/claude-opus-4-7",
        max_tokens=100,
        metadata={"user_id": "hello"},
    )
    assert anthropic_route.call_count == 1
    assert anthropic_route.calls.last.request.headers["x-api-key"] == "bad-key"
    assert bedrock_route.call_count == 1
    assert "authorization" in bedrock_route.calls.last.request.headers
    _assert_anthropic_message(response, _BEDROCK_SONNET)
    assert response["id"] == "msg_bedrock"


@pytest.mark.asyncio
async def test_router_aanthropic_messages_bedrock_converse_and_invoke(respx_mock, monkeypatch):
    _set_fake_aws_env(monkeypatch)
    converse_route: Final = respx_mock.post(url__regex=_BEDROCK_CONVERSE_URL).mock(
        return_value=httpx.Response(200, json=_CONVERSE_BODY)
    )
    invoke_route: Final = respx_mock.post(url__regex=_BEDROCK_INVOKE_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_body(model=_BEDROCK_SONNET))
    )
    converse_model: Final = f"bedrock/converse/{_BEDROCK_SONNET}"
    invoke_model: Final = f"bedrock/{_BEDROCK_SONNET}"
    router: Final = Router(
        model_list=[
            {"model_name": converse_model, "litellm_params": {"model": converse_model}},
            {"model_name": invoke_model, "litellm_params": {"model": invoke_model}},
        ]
    )
    converse_response: Final = await router.aanthropic_messages(
        messages=[{"role": "user", "content": _PROMPT}], model=converse_model, max_tokens=100
    )
    invoke_response: Final = await router.aanthropic_messages(
        messages=[{"role": "user", "content": _PROMPT}], model=invoke_model, max_tokens=100
    )
    assert converse_route.call_count == 1
    assert invoke_route.call_count == 1
    converse_request: Final = converse_route.calls.last.request
    invoke_request: Final = invoke_route.calls.last.request
    assert "authorization" in converse_request.headers
    assert "authorization" in invoke_request.headers
    assert json.loads(converse_request.read())["messages"] == [{"role": "user", "content": [{"text": _PROMPT}]}]
    assert json.loads(invoke_request.read())["messages"] == [{"role": "user", "content": _PROMPT}]
    _assert_anthropic_message(converse_response, _BEDROCK_SONNET)
    _assert_anthropic_message(invoke_response, _BEDROCK_SONNET)


def test_sync_openai_bridge_anthropic_messages_returns_content_blocks(respx_mock):
    route: Final = respx_mock.post(url__regex=_OPENAI_RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=_OPENAI_BODY)
    )
    response: Final = litellm.anthropic.messages.create(
        messages=[{"role": "user", "content": _PROMPT}],
        model=_OPENAI_MODEL,
        max_tokens=100,
        api_key="fake-key",
    )
    sent: Final = json.loads(route.calls.last.request.read())
    assert sent["model"] == "gpt-4.1-mini"
    assert isinstance(response, dict)
    assert response["content"][0]["text"] == _JOKE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "url", "body", "expected_model"),
    [
        pytest.param(_ANTHROPIC_MODEL, _ANTHROPIC_URL, _anthropic_body(), _ANTHROPIC_MODEL, id="anthropic"),
        pytest.param(
            _BEDROCK_MODEL,
            _BEDROCK_INVOKE_URL,
            _anthropic_body(model="us.anthropic.claude-haiku-4-5-20251001-v1:0"),
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            id="bedrock-invoke",
        ),
        pytest.param(_OPENAI_MODEL, _OPENAI_RESPONSES_URL, _OPENAI_BODY, "gpt-4.1-mini", id="openai-bridge"),
    ],
)
async def test_acreate_non_streaming_returns_dict_content_blocks(
    respx_mock, monkeypatch, model: str, url: str, body: Mapping[str, object], expected_model: str
):
    _set_fake_aws_env(monkeypatch)
    route: Final = respx_mock.post(url__regex=url).mock(return_value=httpx.Response(200, json=body))
    response: Final = await litellm.anthropic.messages.acreate(
        messages=[{"role": "user", "content": _PROMPT}],
        model=model,
        max_tokens=100,
        api_key="fake-key",
    )
    assert route.call_count == 1
    _assert_anthropic_message(response, expected_model)


@pytest.mark.asyncio
async def test_router_aanthropic_messages_non_streaming_logs_usage_model_and_cost(respx_mock, monkeypatch):
    messages: Final = _unique_messages()
    recorder: Final = _RecordingLogger(messages)
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    respx_mock.post(url__regex=_ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=_anthropic_body()))
    response: Final = await _router(_ALIAS, _ANTHROPIC_MODEL).aanthropic_messages(
        messages=messages, model=_ALIAS, max_tokens=100
    )
    await _wait_for_payload(recorder)
    assert len(recorder.payloads) == 1
    payload: Final = recorder.payloads[0]
    assert payload["status"] == "success"
    assert payload["messages"] == messages
    assert payload["response"] is not None
    assert payload["model"] == _ANTHROPIC_MODEL
    assert payload["model_group"] == _ALIAS
    assert payload["response_cost"] > 0
    assert payload["prompt_tokens"] == response["usage"]["input_tokens"] == 10
    assert payload["completion_tokens"] == response["usage"]["output_tokens"] == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "url", "content_type", "body", "expected_model"),
    [
        pytest.param(
            _ANTHROPIC_MODEL,
            _ANTHROPIC_URL,
            "text/event-stream",
            _anthropic_sse_body(),
            _ANTHROPIC_MODEL,
            id="anthropic",
        ),
        pytest.param(
            _BEDROCK_MODEL,
            _BEDROCK_INVOKE_STREAM_URL,
            "application/vnd.amazon.eventstream",
            _invoke_stream_body(),
            _BEDROCK_MODEL,
            id="bedrock-invoke",
        ),
    ],
)
async def test_router_aanthropic_messages_streaming_logs_usage_model_and_cost(
    respx_mock, monkeypatch, model: str, url: str, content_type: str, body: bytes, expected_model: str
):
    _set_fake_aws_env(monkeypatch)
    messages: Final = _unique_messages()
    recorder: Final = _RecordingLogger(messages)
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    respx_mock.post(url__regex=url).mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": content_type})
    )
    stream: Final = await _router(_ALIAS, model).aanthropic_messages(
        messages=messages, model=_ALIAS, max_tokens=100, stream=True
    )
    events: Final = await _stream_events(stream)
    usages: Final = tuple(
        event["usage"] if "usage" in event else event["message"]["usage"]
        for event in events
        if "usage" in event or (event.get("type") == "message_start" and "usage" in event["message"])
    )
    assert usages, events
    await _wait_for_payload(recorder)
    assert len(recorder.payloads) == 1
    payload: Final = recorder.payloads[0]
    assert payload["status"] == "success"
    assert payload["messages"] == messages
    assert payload["response"] is not None
    assert payload["model"] == expected_model
    assert payload["response_cost"] > 0
    assert payload["prompt_tokens"] == max(usage.get("input_tokens", 0) for usage in usages) == 10
    assert payload["completion_tokens"] == max(usage.get("output_tokens", 0) for usage in usages) == 20


_LARGE_SYSTEM_PROMPT: Final = "This is a comprehensive legal agreement between Party A and Party B. " * 100


def _cached_system() -> list[AnthropicMessagesTextParam]:
    return [{"type": "text", "text": _LARGE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]


@pytest.mark.asyncio
async def test_bedrock_converse_system_prompt_caching_returns_cache_tokens(respx_mock, monkeypatch):
    _set_fake_aws_env(monkeypatch)
    route: Final = respx_mock.post(url__regex=_BEDROCK_CONVERSE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                **_CONVERSE_BODY,
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 20,
                    "totalTokens": 580,
                    "cacheReadInputTokens": 500,
                    "cacheWriteInputTokens": 50,
                },
            },
        )
    )
    response: Final = await litellm.anthropic.messages.acreate(
        model=f"bedrock/converse/{_BEDROCK_SONNET}",
        messages=[{"role": "user", "content": "What are the key terms?"}],
        system=_cached_system(),
        max_tokens=100,
    )
    sent: Final = json.loads(route.calls.last.request.read())
    assert sent["system"] == [{"text": _LARGE_SYSTEM_PROMPT}, {"cachePoint": {"type": "default"}}]
    assert isinstance(response, dict)
    assert response["usage"]["cache_creation_input_tokens"] == 50
    assert response["usage"]["cache_read_input_tokens"] == 500


@pytest.mark.asyncio
async def test_bedrock_invoke_system_prompt_caching_returns_cache_tokens(respx_mock, monkeypatch):
    _set_fake_aws_env(monkeypatch)
    route: Final = respx_mock.post(url__regex=_BEDROCK_INVOKE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                **_anthropic_body(model=_BEDROCK_SONNET),
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 50,
                    "cache_read_input_tokens": 500,
                },
            },
        )
    )
    response: Final = await litellm.anthropic.messages.acreate(
        model=f"bedrock/invoke/{_BEDROCK_SONNET}",
        messages=[{"role": "user", "content": "What are the key terms?"}],
        system=_cached_system(),
        max_tokens=100,
    )
    sent: Final = json.loads(route.calls.last.request.read())
    assert sent["system"] == _cached_system()
    assert isinstance(response, dict)
    assert response["usage"]["cache_creation_input_tokens"] == 50
    assert response["usage"]["cache_read_input_tokens"] == 500


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "url", "body"),
    [
        pytest.param(
            f"bedrock/converse/{_BEDROCK_SONNET}", _BEDROCK_CONVERSE_STREAM_URL, _converse_stream_body(), id="converse"
        ),
        pytest.param(
            f"bedrock/invoke/{_BEDROCK_SONNET}", _BEDROCK_INVOKE_STREAM_URL, _invoke_stream_body(), id="invoke"
        ),
    ],
)
async def test_bedrock_streaming_message_start_carries_cache_usage_fields(
    respx_mock, monkeypatch, model: str, url: str, body: bytes
):
    _set_fake_aws_env(monkeypatch)
    route: Final = respx_mock.post(url__regex=url).mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/vnd.amazon.eventstream"})
    )
    stream: Final = await litellm.anthropic.messages.acreate(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _LARGE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "What are the payment terms in this agreement?"},
                ],
            }
        ],
        max_tokens=100,
        stream=True,
    )
    events: Final = await _stream_events(stream)
    assert route.call_count == 1
    message_starts: Final = [event for event in events if event.get("type") == "message_start"]
    assert len(message_starts) == 1, events
    usage: Final = message_starts[0]["message"]["usage"]
    assert "cache_creation_input_tokens" in usage, usage
    assert "cache_read_input_tokens" in usage, usage


def _tool_search_tools() -> list[AnthropicToolSearchToolRegex | AnthropicMessagesTool]:
    def deferred(name: str, description: str, field: str) -> AnthropicMessagesTool:
        return {
            "name": name,
            "description": description,
            "input_schema": {"type": "object", "properties": {field: {"type": "string"}}, "required": [field]},
            "defer_loading": True,
        }

    return [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        deferred("get_weather", "Get the current weather for a location", "location"),
        deferred("get_stock_price", "Get the current stock price for a ticker symbol", "ticker"),
        deferred("search_web", "Search the web for information", "query"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "tool_name", "tool_input"),
    [
        pytest.param(
            "I need to know the current weather in New York City. Please use the appropriate tool.",
            "get_weather",
            {"location": "New York, NY"},
            id="discovers-weather-tool",
        ),
        pytest.param(
            "What's the stock price of Apple (AAPL)?", "get_stock_price", {"ticker": "AAPL"}, id="multiple-deferred"
        ),
    ],
)
async def test_tool_search_forwards_deferred_tools_and_beta_header(
    respx_mock, prompt: str, tool_name: str, tool_input: Mapping[str, str]
):
    route: Final = respx_mock.post(url__regex=_ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                **_anthropic_body(model="claude-sonnet-4-5-20250929", msg_id="msg_tool"),
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": tool_input},
                ],
                "stop_reason": "tool_use",
            },
        )
    )
    response: Final = await litellm.anthropic.messages.acreate(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": prompt}],
        tools=[dict(tool) for tool in _tool_search_tools()],
        max_tokens=1024,
        api_key="fake-key",
        extra_headers={"anthropic-beta": "advanced-tool-use-2025-11-20"},
    )
    request: Final = route.calls.last.request
    assert "advanced-tool-use-2025-11-20" in request.headers["anthropic-beta"].split(",")
    assert json.loads(request.read())["tools"] == _tool_search_tools()
    assert isinstance(response, dict)
    assert response["stop_reason"] == "tool_use"
    assert [block for block in response["content"] if block["type"] == "tool_use"] == [
        {"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": tool_input}
    ]


async def _native_output_config_sent(
    respx_mock: respx.MockRouter, effort: str, model_info: Mapping[str, object] | None, **params: object
) -> object:
    route: Final = respx_mock.post(url__regex=_ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_body("claude-sonnet-4-6"))
    )
    deployment: Final = {
        "model_name": _ALIAS,
        "litellm_params": {"model": "anthropic/claude-sonnet-4-6", "api_key": "fake-key"},
        **({"model_info": model_info} if model_info is not None else {}),
    }
    await Router(model_list=[deployment]).aanthropic_messages(
        messages=[{"role": "user", "content": _PROMPT}],
        model=_ALIAS,
        max_tokens=100,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        **params,
    )
    return json.loads(route.calls.last.request.read()).get("output_config")


@pytest.mark.asyncio
async def test_native_messages_round_output_config_effort_onto_the_deployment_declaration(
    respx_mock: respx.MockRouter,
) -> None:
    declared: Final = {"reasoning_effort_levels": ["low", "high"]}

    assert await _native_output_config_sent(respx_mock, "medium", declared) == {"effort": "high"}
    assert await _native_output_config_sent(respx_mock, "medium", None) == {"effort": "medium"}


@pytest.mark.asyncio
async def test_native_messages_drop_an_undeclared_effort_under_drop_params(respx_mock: respx.MockRouter) -> None:
    declared: Final = {"reasoning_effort_levels": ["low", "high"]}

    assert await _native_output_config_sent(respx_mock, "turbo", declared, drop_params=True) is None
    assert await _native_output_config_sent(respx_mock, "turbo", declared) == {"effort": "turbo"}


_DECLARING_LOW_AND_HIGH: Final = {"reasoning_effort_levels": ["low", "high"]}


async def _native_opus_body_sent(
    respx_mock: respx.MockRouter, model_info: Mapping[str, object] | None, **params: object
) -> Mapping[str, object]:
    route: Final = respx_mock.post(url__regex=_ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_body("claude-opus-5-5"))
    )
    deployment: Final = {
        "model_name": _ALIAS,
        "litellm_params": {"model": "anthropic/claude-opus-5-5", "api_key": "fake-key"},
        **({"model_info": model_info} if model_info is not None else {}),
    }
    await Router(model_list=[deployment]).aanthropic_messages(
        messages=[{"role": "user", "content": _PROMPT}], model=_ALIAS, max_tokens=4096, **params
    )
    return json.loads(route.calls.last.request.read())


@pytest.mark.asyncio
@pytest.mark.parametrize("model_info, expected_effort", [(_DECLARING_LOW_AND_HIGH, "high"), (None, "medium")])
async def test_native_messages_round_a_thinking_budget_on_an_adaptive_claude_deployment(
    respx_mock: respx.MockRouter, model_info: Mapping[str, object] | None, expected_effort: str
) -> None:
    sent: Final = await _native_opus_body_sent(
        respx_mock, model_info, thinking={"type": "enabled", "budget_tokens": 2048, "display": "summarized"}
    )

    assert sent["output_config"] == {"effort": expected_effort}
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}


@pytest.mark.asyncio
@pytest.mark.parametrize("dropped_effort", ["turbo", "none"])
async def test_native_messages_drop_params_leave_the_thinking_budget_to_set_the_effort_beside_a_dropped_level(
    respx_mock: respx.MockRouter, dropped_effort: str
) -> None:
    budget: Final = {"type": "enabled", "budget_tokens": 2048}

    dropped: Final = await _native_opus_body_sent(
        respx_mock,
        _DECLARING_LOW_AND_HIGH,
        thinking=budget,
        output_config={"effort": dropped_effort, "format": {"type": "json_schema", "schema": {"type": "object"}}},
        drop_params=True,
    )
    budget_alone: Final = await _native_opus_body_sent(
        respx_mock,
        _DECLARING_LOW_AND_HIGH,
        thinking=budget,
        output_config={"format": {"type": "json_schema", "schema": {"type": "object"}}},
        drop_params=True,
    )

    assert budget_alone["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert dropped == budget_alone


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [["none"], []])
async def test_native_messages_drop_params_drop_a_budget_effort_the_deployment_does_not_declare(
    respx_mock: respx.MockRouter, declared: list[str]
) -> None:
    model_info: Final = {"reasoning_effort_levels": declared}

    sent: Final = await _native_opus_body_sent(
        respx_mock,
        model_info,
        thinking={"type": "enabled", "budget_tokens": 2048, "display": "summarized"},
        drop_params=True,
    )

    assert "output_config" not in sent
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", [False, True])
async def test_native_messages_send_no_effort_for_a_default_effort_beside_a_thinking_budget(
    respx_mock: respx.MockRouter, drop_params: bool
) -> None:
    sent: Final = await _native_opus_body_sent(
        respx_mock,
        _DECLARING_LOW_AND_HIGH,
        thinking={"type": "enabled", "budget_tokens": 2048},
        output_config={"effort": "default"},
        drop_params=drop_params,
    )

    assert "output_config" not in sent
    assert sent["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
@pytest.mark.parametrize("model_info", [None, _DECLARING_LOW_AND_HIGH])
@pytest.mark.parametrize("drop_globally", [False, True])
async def test_native_messages_drop_params_drops_a_non_numeric_thinking_budget_on_adaptive_claude(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    model_info: Mapping[str, object] | None,
    drop_globally: bool,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", drop_globally)
    request_drop: Final = {} if drop_globally else {"drop_params": True}

    dropped: Final = await _native_opus_body_sent(
        respx_mock, model_info, thinking={"type": "enabled", "budget_tokens": "lots"}, **request_drop
    )
    adaptive_alone: Final = await _native_opus_body_sent(
        respx_mock, model_info, thinking={"type": "adaptive"}, **request_drop
    )

    assert "output_config" not in adaptive_alone
    assert dropped == adaptive_alone


_MIXED_GROUP: Final = "gpt-5.6-mixed"
_MIXED_AZURE_API_BASE: Final = "https://mixed-group.openai.azure.com"
_MIXED_AZURE_CHAT_URL: Final = r".*mixed-group\.openai\.azure\.com/openai/deployments/.*/chat/completions.*"
_MIXED_DEPLOYMENT_IDS: Final = ("mixed-openai-responses-bridge", "mixed-azure-chat-bridge")
_MIXED_AZURE_CHAT_BODY: Final = {
    "id": "chatcmpl-mixed",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-5.6",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": _JOKE}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}


def _mixed_openai_azure_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": _MIXED_GROUP,
                "litellm_params": {"model": "openai/gpt-5.6", "api_key": "fake-key"},
                "model_info": {"id": _MIXED_DEPLOYMENT_IDS[0]},
            },
            {
                "model_name": _MIXED_GROUP,
                "litellm_params": {"model": "azure/gpt-5.6", "api_key": "fake-key", "api_base": _MIXED_AZURE_API_BASE},
                "model_info": {"id": _MIXED_DEPLOYMENT_IDS[1]},
            },
        ],
        num_retries=0,
    )


async def _null_tool_result_text_rejection(router: Router, deployment_id: str) -> litellm.BadRequestError:
    with pytest.raises(litellm.BadRequestError) as rejected:
        await router.aanthropic_messages(
            model=deployment_id,
            max_tokens=64,
            messages=[
                {"role": "user", "content": "run the build"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_build", "name": "bash", "input": {"cmd": "make"}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_build",
                            "content": [{"type": "text", "text": "build ok"}, {"type": "text", "text": None}],
                        }
                    ],
                },
            ],
        )
    return rejected.value


@pytest.mark.asyncio
async def test_mixed_openai_azure_group_answers_the_same_400_for_a_null_tool_result_text_on_either_deployment(
    respx_mock,
):
    openai_route: Final = respx_mock.post(url__regex=_OPENAI_RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=_OPENAI_BODY)
    )
    azure_route: Final = respx_mock.post(url__regex=_MIXED_AZURE_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_MIXED_AZURE_CHAT_BODY)
    )
    router: Final = _mixed_openai_azure_router()

    rejections: Final = [
        await _null_tool_result_text_rejection(router, deployment_id) for deployment_id in _MIXED_DEPLOYMENT_IDS
    ]

    assert [(rejection.status_code, rejection.llm_provider) for rejection in rejections] == [
        (400, "openai"),
        (400, "azure"),
    ]
    assert {rejection.message.partition("\n\n")[0] for rejection in rejections} == {
        "litellm.BadRequestError: tool_result toolu_build content.1.text: Input should be a valid string"
    }
    assert openai_route.call_count == 0
    assert azure_route.call_count == 0
