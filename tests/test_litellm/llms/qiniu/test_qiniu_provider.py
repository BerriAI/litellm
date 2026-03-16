"""
Tests for Qiniu provider
"""

import json
import math

import pytest
import respx

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token


@pytest.fixture
def qiniu_response():
    return {
        "id": "chatcmpl-qiniu-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "deepseek/deepseek-v3.1-terminus-thinking",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.fixture
def qiniu_stream_chunks():
    """SSE chunks for streaming tests"""
    return [
        b'data: {"id":"chatcmpl-qiniu-stream","object":"chat.completion.chunk","created":1677652288,"model":"DeepSeek-V3.1-Terminus-Thinking","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-qiniu-stream","object":"chat.completion.chunk","created":1677652288,"model":"DeepSeek-V3.1-Terminus-Thinking","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-qiniu-stream","object":"chat.completion.chunk","created":1677652288,"model":"DeepSeek-V3.1-Terminus-Thinking","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]


def test_get_llm_provider_qiniu():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        "qiniu/deepseek/deepseek-v3.1-terminus-thinking"
    )
    assert model == "deepseek/deepseek-v3.1-terminus-thinking"
    assert provider == "qiniu"
    assert api_base == "https://api.qnaigc.com/v1"


def test_qiniu_in_provider_lists():
    assert "qiniu" in litellm.openai_compatible_providers
    assert "qiniu" in litellm.provider_list


def test_qiniu_models_in_model_cost():
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_key = "qiniu/deepseek/deepseek-v3.1-terminus-thinking"
    assert model_key in litellm.model_cost, f"{model_key} not found in model_cost"
    assert litellm.model_cost[model_key]["litellm_provider"] == "qiniu"


def test_qiniu_cost_calculation():
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    prompt_cost, completion_cost = cost_per_token(
        model="qiniu/deepseek/deepseek-v3.1-terminus-thinking",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    # $0.4/M input, $1.2/M output
    assert math.isclose(prompt_cost, 0.4, rel_tol=1e-6)
    assert math.isclose(completion_cost, 1.2, rel_tol=1e-6)


@pytest.mark.asyncio
async def test_qiniu_acompletion(respx_mock, qiniu_response, monkeypatch):
    monkeypatch.setenv("QINIU_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.qnaigc.com/v1/chat/completions").respond(
        json=qiniu_response
    )

    response = await litellm.acompletion(
        model="qiniu/deepseek/deepseek-v3.1-terminus-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you?"
    assert response.usage.total_tokens == 25

    request = respx_mock.calls[0].request
    assert "api.qnaigc.com" in str(request.url)
    assert request.headers["Authorization"] == "Bearer test-api-key"

    # Verify param_mappings: max_tokens sent (not max_completion_tokens)
    body = json.loads(request.content)
    assert body["max_tokens"] == 20
    assert "max_completion_tokens" not in body


def test_qiniu_sync_completion(respx_mock, qiniu_response, monkeypatch):
    monkeypatch.setenv("QINIU_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.qnaigc.com/v1/chat/completions").respond(
        json=qiniu_response
    )

    response = completion(
        model="qiniu/deepseek/deepseek-v3.1-terminus-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you?"
    assert response.usage.total_tokens == 25

    # Verify param_mappings: max_tokens sent (not max_completion_tokens)
    request = respx_mock.calls[0].request
    body = json.loads(request.content)
    assert body["max_tokens"] == 20
    assert "max_completion_tokens" not in body


def test_qiniu_sync_streaming(respx_mock, qiniu_stream_chunks, monkeypatch):
    monkeypatch.setenv("QINIU_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.qnaigc.com/v1/chat/completions").respond(
        content=b"".join(qiniu_stream_chunks),
        headers={"Content-Type": "text/event-stream"},
    )

    response = completion(
        model="qiniu/deepseek/deepseek-v3.1-terminus-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )

    chunks = list(response)
    assert len(chunks) > 0
    content = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chunks
        if chunk.choices[0].delta.content
    )
    assert content == "Hello!"


@pytest.mark.asyncio
async def test_qiniu_async_streaming(respx_mock, qiniu_stream_chunks, monkeypatch):
    monkeypatch.setenv("QINIU_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.qnaigc.com/v1/chat/completions").respond(
        content=b"".join(qiniu_stream_chunks),
        headers={"Content-Type": "text/event-stream"},
    )

    response = await litellm.acompletion(
        model="qiniu/deepseek/deepseek-v3.1-terminus-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )

    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) > 0
    content = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chunks
        if chunk.choices[0].delta.content
    )
    assert content == "Hello!"
