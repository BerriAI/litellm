import asyncio
import json
import time

import httpx
import pytest

import litellm

OPENAI_API_BASE = "https://example.openai.test/v1"


def _completion_payload(response_id="chatcmpl-test"):
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _stream_payload(response_id="chatcmpl-stream"):
    chunks = [
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    ).encode()


def _mock_openai_completion_transport(
    monkeypatch, *, stream=False, response_id="chatcmpl-test"
):
    from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport

    calls = {"count": 0}

    async def delayed_response(_transport, request):
        calls["count"] += 1
        await asyncio.sleep(0.2)
        if stream:
            return httpx.Response(
                200,
                content=_stream_payload(response_id),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx.Response(
            200, json=_completion_payload(response_id), request=request
        )

    monkeypatch.setattr(
        LiteLLMAiohttpTransport,
        "handle_async_request",
        delayed_response,
    )
    return calls


def _assert_overhead_is_smaller_than_total(response, total_time_ms):
    litellm_overhead_ms = response._hidden_params["litellm_overhead_time_ms"]
    overhead_percent = litellm_overhead_ms * 100 / total_time_ms

    assert litellm_overhead_ms > 0
    assert litellm_overhead_ms < 1000
    assert litellm_overhead_ms < total_time_ms
    assert overhead_percent < 40


@pytest.fixture(autouse=True)
def reset_litellm_state():
    litellm.cache = None
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []
    yield
    litellm.cache = None
    litellm.callbacks = []


@pytest.mark.asyncio
async def test_litellm_overhead_non_streaming(monkeypatch):
    calls = _mock_openai_completion_transport(
        monkeypatch, response_id="chatcmpl-non-stream"
    )

    start_time = time.perf_counter()
    response = await litellm.acompletion(
        model="gpt-4o",
        api_key="test-key",
        api_base=OPENAI_API_BASE,
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    total_time_ms = (time.perf_counter() - start_time) * 1000

    assert calls["count"] == 1
    _assert_overhead_is_smaller_than_total(response, total_time_ms)


@pytest.mark.asyncio
async def test_litellm_overhead_stream(monkeypatch):
    calls = _mock_openai_completion_transport(
        monkeypatch, stream=True, response_id="chatcmpl-stream"
    )

    start_time = time.perf_counter()
    response = await litellm.acompletion(
        model="gpt-4o",
        api_key="test-key",
        api_base=OPENAI_API_BASE,
        messages=[{"role": "user", "content": "Hello, world!"}],
        stream=True,
    )

    async for _chunk in response:
        pass

    total_time_ms = (time.perf_counter() - start_time) * 1000

    assert calls["count"] == 1
    _assert_overhead_is_smaller_than_total(response, total_time_ms)


@pytest.mark.asyncio
async def test_litellm_overhead_cache_hit(monkeypatch):
    from litellm.caching.caching import Cache

    calls = _mock_openai_completion_transport(monkeypatch, response_id="chatcmpl-cache")
    litellm.cache = Cache()

    messages = [{"role": "user", "content": "Hello, world! Cache test"}]
    response1 = await litellm.acompletion(
        model="gpt-4o",
        api_key="test-key",
        api_base=OPENAI_API_BASE,
        messages=messages,
        caching=True,
    )
    await asyncio.sleep(0.5)
    response2 = await litellm.acompletion(
        model="gpt-4o",
        api_key="test-key",
        api_base=OPENAI_API_BASE,
        messages=messages,
        caching=True,
    )

    assert calls["count"] == 1
    assert response1.id == response2.id
    assert "_response_ms" in response2._hidden_params
    assert response2._hidden_params["litellm_overhead_time_ms"] > 0
    assert (
        response2._hidden_params["litellm_overhead_time_ms"]
        < response2._hidden_params["_response_ms"]
    )
