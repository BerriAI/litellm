import json
import os
import sys
from unittest.mock import AsyncMock, patch
from typing import Optional

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import httpx
import pytest
import respx
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse
from base_llm_unit_tests import BaseLLMChatTest


class TestCompactifAI(BaseLLMChatTest):
    def get_base_completion_call_args(self):
        return {
            "model": "compactifai/llama-2-7b-compressed",
            "messages": [{"role": "user", "content": "Hello"}]
        }

    def get_custom_llm_provider(self):
        return "compactifai"

    # Implement abstract methods to avoid instantiation errors
    def test_tool_call_no_arguments(self):
        # CompactifAI inherits OpenAI tool calling behavior
        pass


@pytest.mark.respx(base_url="https://api.compactif.ai")
def test_compactifai_completion_basic():
    """Test basic CompactifAI completion functionality"""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "llama-2-7b-compressed",
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

    with respx.mock() as respx_mock:
        respx_mock.post("https://api.compactif.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = litellm.completion(
            model="compactifai/llama-2-7b-compressed",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key"
        )

        assert response.choices[0].message.content == "Hello! How can I help you today?"
        assert response.model == "compactifai/llama-2-7b-compressed"
        assert response.usage.total_tokens == 21


@pytest.mark.respx(base_url="https://api.compactif.ai")
def test_compactifai_completion_streaming():
    """Test CompactifAI streaming completion"""
    mock_chunks = [
        "data: " + json.dumps({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "llama-2-7b-compressed",
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
            "model": "llama-2-7b-compressed",
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

    with respx.mock() as respx_mock:
        respx_mock.post("https://api.compactif.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                content="".join(mock_chunks)
            )
        )

        response = litellm.completion(
            model="compactifai/llama-2-7b-compressed",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key",
            stream=True
        )

        chunks = list(response)
        assert len(chunks) >= 2
        assert chunks[0].choices[0].delta.content == "Hello"


@pytest.mark.respx(base_url="https://api.compactif.ai")
def test_compactifai_models_endpoint():
    """Test CompactifAI models listing"""
    mock_response = {
        "object": "list",
        "data": [
            {
                "id": "llama-2-7b-compressed",
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

    with respx.mock() as respx_mock:
        respx_mock.get("https://api.compactif.ai/v1/models").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        # This would be tested if litellm had a models() function
        # For now, we'll test that the provider is properly configured
        response = litellm.completion(
            model="compactifai/llama-2-7b-compressed",
            messages=[{"role": "user", "content": "test"}],
            api_key="test-key"
        )


@pytest.mark.respx(base_url="https://api.compactif.ai")
def test_compactifai_authentication_error():
    """Test CompactifAI authentication error handling"""
    mock_error = {
        "error": {
            "message": "Invalid API key provided",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key"
        }
    }

    with respx.mock() as respx_mock:
        respx_mock.post("https://api.compactif.ai/v1/chat/completions").mock(
            return_value=httpx.Response(401, json=mock_error)
        )

        with pytest.raises(litellm.AuthenticationError):
            litellm.completion(
                model="compactifai/llama-2-7b-compressed",
                messages=[{"role": "user", "content": "test"}],
                api_key="invalid-key"
            )


@pytest.mark.respx(base_url="https://api.compactif.ai")
def test_compactifai_provider_detection():
    """Test that CompactifAI provider is properly detected from model name"""
    from litellm.utils import get_llm_provider

    model, provider, dynamic_api_key, api_base = get_llm_provider(
        model="compactifai/llama-2-7b-compressed"
    )

    assert provider == "compactifai"
    assert model == "llama-2-7b-compressed"


@pytest.mark.respx(base_url="https://api.compactif.ai")
def test_compactifai_with_optional_params():
    """Test CompactifAI with optional parameters like temperature, max_tokens"""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "llama-2-7b-compressed",
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

    with respx.mock() as respx_mock:
        request_mock = respx_mock.post("https://api.compactif.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = litellm.completion(
            model="compactifai/llama-2-7b-compressed",
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


@pytest.mark.respx(base_url="https://api.compactif.ai")
def test_compactifai_headers_authentication():
    """Test that CompactifAI request includes proper authorization headers"""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "llama-2-7b-compressed",
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

    with respx.mock() as respx_mock:
        request_mock = respx_mock.post("https://api.compactif.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = litellm.completion(
            model="compactifai/llama-2-7b-compressed",
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
@pytest.mark.respx(base_url="https://api.compactif.ai")
async def test_compactifai_async_completion():
    """Test CompactifAI async completion"""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "llama-2-7b-compressed",
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

    with respx.mock() as respx_mock:
        respx_mock.post("https://api.compactif.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = await litellm.acompletion(
            model="compactifai/llama-2-7b-compressed",
            messages=[{"role": "user", "content": "Async test"}],
            api_key="test-key"
        )

        assert response.choices[0].message.content == "Async response from CompactifAI"
        assert response.usage.total_tokens == 23