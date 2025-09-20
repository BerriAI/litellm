import json
import os
import sys
from unittest.mock import AsyncMock, patch
from typing import Optional

import httpx
import pytest
import respx
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse


@pytest.mark.respx()
def test_compactifai_completion_basic(respx_mock):
    """Test basic CompactifAI completion functionality"""
    litellm.disable_aiohttp_transport = True

    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "cai-llama-3-1-8b-slim",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }

    respx_mock.post("https://api.compactif.ai/v1/chat/completions").respond(
        json=mock_response, status_code=200
    )

    response = litellm.completion(
        model="compactifai/cai-llama-3-1-8b-slim",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="test-key"
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.model == "compactifai/cai-llama-3-1-8b-slim"
    assert response.usage.total_tokens == 21


@pytest.mark.respx()
def test_compactifai_completion_streaming(respx_mock):
    """Test CompactifAI streaming completion"""
    litellm.disable_aiohttp_transport = True

    mock_chunks = [
        "data: " + json.dumps({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "cai-llama-3-1-8b-slim",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None
                }
            ]
        }) + "\n\n",
        "data: " + json.dumps({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "cai-llama-3-1-8b-slim",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "!"},
                    "finish_reason": "stop"
                }
            ]
        }) + "\n\n",
        "data: [DONE]\n\n"
    ]

    respx_mock.post("https://api.compactif.ai/v1/chat/completions").respond(
        status_code=200,
        headers={"content-type": "text/plain"},
        content="".join(mock_chunks)
    )

    response = litellm.completion(
        model="compactifai/cai-llama-3-1-8b-slim",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="test-key",
        stream=True
    )

    chunks = list(response)
    assert len(chunks) >= 2
    assert chunks[0].choices[0].delta.content == "Hello"


@pytest.mark.respx()
def test_compactifai_models_endpoint(respx_mock):
    """Test CompactifAI models listing"""
    litellm.disable_aiohttp_transport = True

    mock_response = {
        "object": "list",
        "data": [
            {
                "id": "cai-llama-3-1-8b-slim",
                "object": "model",
                "created": 1677610602,
                "owned_by": "compactifai"
            },
            {
                "id": "mistral-7b-compressed",
                "object": "model",
                "created": 1677610602,
                "owned_by": "compactifai"
            }
        ]
    }

    respx_mock.post("https://api.compactif.ai/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "cai-llama-3-1-8b-slim",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15
            }
        },
        status_code=200
    )

    # This would be tested if litellm had a models() function
    # For now, we'll test that the provider is properly configured
    response = litellm.completion(
        model="compactifai/cai-llama-3-1-8b-slim",
        messages=[{"role": "user", "content": "test"}],
        api_key="test-key"
    )


@pytest.mark.respx()
def test_compactifai_authentication_error(respx_mock):
    """Test CompactifAI authentication error handling"""
    litellm.disable_aiohttp_transport = True

    mock_error = {
        "error": {
            "message": "Invalid API key provided",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key"
        }
    }

    respx_mock.post("https://api.compactif.ai/v1/chat/completions").respond(
        json=mock_error, status_code=401
    )

    with pytest.raises(litellm.APIConnectionError) as exc_info:
        litellm.completion(
            model="compactifai/cai-llama-3-1-8b-slim",
            messages=[{"role": "user", "content": "test"}],
            api_key="invalid-key"
        )

    # Verify the error contains the expected authentication error message
    assert "Invalid API key provided" in str(exc_info.value)


@pytest.mark.respx()
def test_compactifai_provider_detection(respx_mock):
    """Test that CompactifAI provider is properly detected from model name"""
    from litellm.utils import get_llm_provider

    model, provider, dynamic_api_key, api_base = get_llm_provider(
        model="compactifai/cai-llama-3-1-8b-slim"
    )

    assert provider == "compactifai"
    assert model == "cai-llama-3-1-8b-slim"


@pytest.mark.respx()
def test_compactifai_with_optional_params(respx_mock):
    """Test CompactifAI with optional parameters like temperature, max_tokens"""
    litellm.disable_aiohttp_transport = True

    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "cai-llama-3-1-8b-slim",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response with custom parameters."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 20,
            "total_tokens": 35
        }
    }

    request_mock = respx_mock.post("https://api.compactif.ai/v1/chat/completions").respond(
        json=mock_response, status_code=200
    )

    response = litellm.completion(
        model="compactifai/cai-llama-3-1-8b-slim",
        messages=[{"role": "user", "content": "Hello with params"}],
        api_key="test-key",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9
    )

    assert response.choices[0].message.content == "This is a test response with custom parameters."

    # Verify the request was made with correct parameters
    assert request_mock.called
    request_data = request_mock.calls[0].request.content
    parsed_data = json.loads(request_data)
    assert parsed_data["temperature"] == 0.7
    assert parsed_data["max_tokens"] == 100
    assert parsed_data["top_p"] == 0.9


@pytest.mark.respx()
def test_compactifai_headers_authentication(respx_mock):
    """Test that CompactifAI request includes proper authorization headers"""
    litellm.disable_aiohttp_transport = True

    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "cai-llama-3-1-8b-slim",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15
        }
    }

    request_mock = respx_mock.post("https://api.compactif.ai/v1/chat/completions").respond(
        json=mock_response, status_code=200
    )

    response = litellm.completion(
        model="compactifai/cai-llama-3-1-8b-slim",
        messages=[{"role": "user", "content": "Test auth"}],
        api_key="test-api-key-123"
    )

    assert response.choices[0].message.content == "Test response"

    # Verify authorization header was set correctly
    assert request_mock.called
    request_headers = request_mock.calls[0].request.headers
    assert "authorization" in request_headers
    assert request_headers["authorization"] == "Bearer test-api-key-123"


@pytest.mark.asyncio
@pytest.mark.respx()
async def test_compactifai_async_completion(respx_mock):
    """Test CompactifAI async completion"""
    litellm.disable_aiohttp_transport = True

    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "cai-llama-3-1-8b-slim",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Async response from CompactifAI"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 15,
            "total_tokens": 23
        }
    }

    respx_mock.post("https://api.compactif.ai/v1/chat/completions").respond(
        json=mock_response, status_code=200
    )

    response = await litellm.acompletion(
        model="compactifai/cai-llama-3-1-8b-slim",
        messages=[{"role": "user", "content": "Async test"}],
        api_key="test-key"
    )

    assert response.choices[0].message.content == "Async response from CompactifAI"
    assert response.usage.total_tokens == 23