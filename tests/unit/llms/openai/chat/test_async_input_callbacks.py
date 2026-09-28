import asyncio
from typing import Final

import httpx
import pytest
import respx

import litellm


def _completion_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-async-input-callback",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.mark.asyncio
@respx.mock
async def test_acompletion_awaits_async_input_callbacks_before_provider(monkeypatch):
    events: Final[list[str]] = []
    callback_payloads: Final[list[dict]] = []

    async def failing_callback(kwargs):
        events.append("failing_callback")
        raise RuntimeError("input callback failed")

    async def succeeding_callback(kwargs):
        events.append("callback_started")
        await asyncio.sleep(0)
        callback_payloads.append(kwargs["additional_args"]["complete_input_dict"])
        events.append("callback_finished")

    def provider_request(request):
        events.append("provider")
        return _completion_response()

    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "_async_input_callback", [])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.logging_callback_manager.add_litellm_input_callback(failing_callback)
    litellm.logging_callback_manager.add_litellm_input_callback(succeeding_callback)
    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=provider_request)

    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
    )

    assert response.choices[0].message.content == "ok"
    assert events == ["failing_callback", "callback_started", "callback_finished", "provider"]
    assert callback_payloads[0]["model"] == "gpt-4o-mini"
    assert callback_payloads[0]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
@respx.mock
async def test_streaming_acompletion_awaits_async_input_callback_before_provider(monkeypatch):
    events: Final[list[str]] = []

    async def callback(kwargs):
        await asyncio.sleep(0)
        events.append("callback")

    def provider_request(request):
        events.append("provider")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1,'
                b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"ok"},'
                b'"finish_reason":null}]}\n\ndata: [DONE]\n\n'
            ),
        )

    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "_async_input_callback", [])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.logging_callback_manager.add_litellm_input_callback(callback)
    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=provider_request)

    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        stream=True,
    )

    assert events == ["callback", "provider"]
    await response.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_acompletion_cache_hit_does_not_run_input_callbacks(monkeypatch):
    callback_calls: Final[list[int]] = []

    async def callback(kwargs):
        callback_calls.append(1)

    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "_async_input_callback", [])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))
    litellm.logging_callback_manager.add_litellm_input_callback(callback)
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(return_value=_completion_response())
    request = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "api_key": "test-key",
        "caching": True,
    }

    await litellm.acompletion(**request)
    await asyncio.sleep(0.1)
    await litellm.acompletion(**request)

    assert route.call_count == 1
    assert callback_calls == [1]


@respx.mock
def test_completion_skips_async_input_callback(monkeypatch):
    callback_calls: Final[list[int]] = []

    async def callback(kwargs):
        callback_calls.append(1)

    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "_async_input_callback", [])
    litellm.logging_callback_manager.add_litellm_input_callback(callback)
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(return_value=_completion_response())

    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
    )

    assert response.choices[0].message.content == "ok"
    assert route.call_count == 1
    assert callback_calls == []
