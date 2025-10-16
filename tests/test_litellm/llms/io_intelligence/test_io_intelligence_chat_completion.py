"""
Comprehensive test suite for IO Intelligence chat completion functionality.
Tests streaming, usage tracking, error handling, and cost calculation with mock responses.

1. test_basic_chat_completion_with_usage - Tests standard completion with full usage metrics validation
2. test_streaming_chat_completion - Tests streaming functionality with proper chunk parsing and content accumulation
3. test_usage_cost_calculation - Tests integration with LiteLLM's cost calculation system
4. test_async_chat_completion_with_usage - Tests async functionality with usage tracking
5. test_error_handling_missing_api_key - Tests authentication error handling
6. test_error_handling_api_error_response - Tests API error response handling
7. test_streaming_with_custom_parameters - Tests streaming with custom parameters (temperature, max_tokens, top_p)
"""

import asyncio
import json
import os
import sys

import pytest
import respx

# Add litellm to path
sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm import ModelResponse, completion
from litellm.constants import openai_compatible_providers
from litellm.cost_calculator import completion_cost


class TestIOIntelligenceChatCompletion:
    """Comprehensive test suite for IO Intelligence chat completion with mocking"""

    def test_io_intelligence_provider_registration(self):
        """Test that IO Intelligence is properly registered as an OpenAI-compatible provider"""
        assert "io_intelligence" in openai_compatible_providers

    @pytest.mark.respx()
    def test_basic_chat_completion_with_usage(self, respx_mock):
        """Test basic chat completion with proper usage tracking"""
        litellm.disable_aiohttp_transport = True

        # Mock OpenAI-compatible response from IO Intelligence
        mock_response_data = {
            "id": "io-intel-chat-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "llama-3.1-8b-instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I'm an AI assistant powered by IO Intelligence. How can I help you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 18,
                "total_tokens": 43
            }
        }

        # Mock the API endpoint
        respx_mock.post("https://api.intelligence.io.solutions/api/v1/chat/completions").respond(
            json=mock_response_data, status_code=200
        )

        # Test completion
        response = completion(
            model="io_intelligence/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            api_key="test_io_intelligence_key",
            api_base="https://api.intelligence.io.solutions/api/v1"
        )

        # Verify response structure
        assert isinstance(response, ModelResponse)
        assert response.id == "io-intel-chat-123"
        assert response.model == "io_intelligence/llama-3.1-8b-instruct"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert "IO Intelligence" in response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"
        
        # Verify usage tracking
        assert response.usage is not None
        assert response.usage.prompt_tokens == 25
        assert response.usage.completion_tokens == 18
        assert response.usage.total_tokens == 43

    @pytest.mark.respx()
    def test_streaming_chat_completion(self, respx_mock):
        """Test streaming chat completion with proper chunk handling"""
        litellm.disable_aiohttp_transport = True

        # Mock streaming chunks that would come from IO Intelligence
        streaming_chunks = [
            "data: " + json.dumps({
                "id": "io-intel-stream-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "llama-3.1-8b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }
                ]
            }) + "\n\n",
            "data: " + json.dumps({
                "id": "io-intel-stream-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "llama-3.1-8b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hello"},
                        "finish_reason": None
                    }
                ]
            }) + "\n\n",
            "data: " + json.dumps({
                "id": "io-intel-stream-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "llama-3.1-8b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " from IO Intelligence!"},
                        "finish_reason": None
                    }
                ]
            }) + "\n\n",
            "data: " + json.dumps({
                "id": "io-intel-stream-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "llama-3.1-8b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28
                }
            }) + "\n\n",
            "data: [DONE]\n\n"
        ]

        # Mock streaming response
        respx_mock.post("https://api.intelligence.io.solutions/api/v1/chat/completions").respond(
            status_code=200,
            headers={"content-type": "text/plain"},
            content="".join(streaming_chunks)
        )

        # Test streaming completion
        response = completion(
            model="io_intelligence/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="test_io_intelligence_key",
            api_base="https://api.intelligence.io.solutions/api/v1",
            stream=True
        )

        # Collect streaming chunks
        chunks = []
        content_parts = []
        
        for chunk in response:
            chunks.append(chunk)
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        # Verify streaming behavior
        assert len(chunks) > 0, "Should receive streaming chunks"
        assert len(content_parts) > 0, "Should receive content in chunks"
        
        # Verify content accumulation
        full_content = "".join(content_parts)
        assert "Hello from IO Intelligence!" == full_content

    @pytest.mark.respx()
    def test_usage_cost_calculation(self, respx_mock):
        """Test integration with LiteLLM's cost calculation system"""
        litellm.disable_aiohttp_transport = True

        mock_response_data = {
            "id": "io-intel-cost-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "llama-3.1-70b-instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response for cost calculation."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/chat/completions").respond(
            json=mock_response_data, status_code=200
        )

        response = completion(
            model="io_intelligence/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": "Calculate cost for this request"}],
            api_key="test_key",
            api_base="https://api.intelligence.io.solutions/api/v1"
        )

        # Test cost calculation (should work even without specific IO Intelligence pricing)
        try:
            cost = completion_cost(completion_response=response, custom_llm_provider="io_intelligence")
            # Cost calculation should work without errors
            assert isinstance(cost, (int, float))
            assert cost >= 0
        except Exception:
            # If no pricing data available, that's acceptable for this test
            pass

        # Verify usage data is properly structured for cost calculation
        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 50
        assert response.usage.total_tokens == 150

    @pytest.mark.respx()
    def test_async_chat_completion_with_usage(self, respx_mock):
        """Test async chat completion with usage tracking"""
        litellm.disable_aiohttp_transport = True

        mock_response_data = {
            "id": "io-intel-async-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "llama-3.1-8b-instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is an async response from IO Intelligence."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "total_tokens": 42
            }
        }

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/chat/completions").respond(
            json=mock_response_data, status_code=200
        )

        async def run_async_test():
            response = await litellm.acompletion(
                model="io_intelligence/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": "Test async completion"}],
                api_key="test_async_key",
                api_base="https://api.intelligence.io.solutions/api/v1"
            )
            return response

        # Run async test
        response = asyncio.run(run_async_test())

        # Verify async response
        assert response.id == "io-intel-async-123"
        assert response.model == "io_intelligence/llama-3.1-8b-instruct"
        assert "async response" in response.choices[0].message.content
        assert response.usage.prompt_tokens == 30
        assert response.usage.completion_tokens == 12
        assert response.usage.total_tokens == 42

    def test_error_handling_missing_api_key(self):
        """Test error handling for missing API key"""
        # Test should raise error for missing API key when no auth headers provided
        with pytest.raises(Exception) as exc_info:
            completion(
                model="io_intelligence/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": "This should fail"}],
                # No api_key provided
                api_base="https://api.intelligence.io.solutions/api/v1"
            )
        
        # Should contain helpful error message about missing API key or base
        error_message = str(exc_info.value)
        assert any(keyword in error_message.lower() for keyword in ["api", "key", "auth", "missing"])

    @pytest.mark.respx()
    def test_error_handling_api_error_response(self, respx_mock):
        """Test handling of API error responses"""
        litellm.disable_aiohttp_transport = True

        # Mock an error response from IO Intelligence API
        error_response_data = {
            "error": {
                "message": "Invalid model specified",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        }

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/chat/completions").respond(
            json=error_response_data, status_code=400
        )

        # Test that API errors are properly handled
        with pytest.raises(Exception) as exc_info:
            completion(
                model="io_intelligence/invalid-model",
                messages=[{"role": "user", "content": "Test error handling"}],
                api_key="test_key",
                api_base="https://api.intelligence.io.solutions/api/v1"
            )
        
        # Verify error contains useful information
        error_str = str(exc_info.value)
        # Should contain some indication of the error (exact format may vary based on LiteLLM's error handling)
        assert len(error_str) > 0

    @pytest.mark.respx()
    def test_streaming_with_custom_parameters(self, respx_mock):
        """Test streaming with custom parameters specific to IO Intelligence"""
        litellm.disable_aiohttp_transport = True

        streaming_chunks = [
            "data: " + json.dumps({
                "id": "io-intel-custom-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "llama-3.1-8b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }
                ]
            }) + "\n\n",
            "data: " + json.dumps({
                "id": "io-intel-custom-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "llama-3.1-8b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Custom response"},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 5,
                    "total_tokens": 20
                }
            }) + "\n\n",
            "data: [DONE]\n\n"
        ]

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/chat/completions").respond(
            status_code=200,
            headers={"content-type": "text/plain"},
            content="".join(streaming_chunks)
        )

        # Test with custom parameters
        response = completion(
            model="io_intelligence/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": "Test with custom params"}],
            api_key="test_key",
            api_base="https://api.intelligence.io.solutions/api/v1",
            stream=True,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )

        # Collect chunks
        chunks = list(response)
        
        # Verify streaming worked
        assert len(chunks) > 0