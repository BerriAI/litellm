"""
Mocked tests for Vertex AI Gemma Models

Maps to: litellm/llms/vertex_ai/vertex_gemma_models/transformation.py
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

import litellm


class TestVertexGemmaCompletion:
    """Test completion flow for Vertex AI Gemma models using litellm.acompletion()"""

    @pytest.mark.asyncio
    async def test_acompletion_basic_request(self):
        """
        Test litellm.acompletion() with Vertex AI Gemma model
        
        Expected URL:
        https://32277599999999999.us-central1-10582012152.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict
        
        Expected Request Body (sent to Vertex):
        {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is machine learning?"
                        }
                    ],
                    "max_tokens": 100
                }
            ]
        }
        
        Expected Vertex Response:
        {
            "deployedModelId": "1207280419999999999",
            "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
            "modelDisplayName": "gemma-3-12b-it-1222199011122",
            "modelVersionId": "1",
            "predictions": {
                "choices": [
                    {
                        "finish_reason": "length",
                        "index": 0,
                        "logprobs": null,
                        "message": {
                            "content": "Okay, let's break down machine learning...",
                            "reasoning_content": null,
                            "role": "assistant",
                            "tool_calls": []
                        },
                        "stop_reason": null
                    }
                ],
                "created": 1759863903,
                "id": "chatcmpl-aaa4288f-2b8e-4bc0-8b14-4e444decd2c4",
                "model": "google/gemma-3-12b-it",
                "object": "chat.completion",
                "prompt_logprobs": null,
                "usage": {
                    "completion_tokens": 100,
                    "prompt_tokens": 14,
                    "prompt_tokens_details": null,
                    "total_tokens": 114
                }
            }
        }
        
        Expected LiteLLM Response: Standard OpenAI format
        """
        # Real Vertex response from user's spec
        mock_vertex_response = {
            "deployedModelId": "1207280419999999999",
            "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
            "modelDisplayName": "gemma-3-12b-it-1222199011122",
            "modelVersionId": "1",
            "predictions": {
                "choices": [
                    {
                        "finish_reason": "length",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": "Okay, let's break down machine learning. Here's a comprehensive explanation, covering the core concepts, types, and some examples, tailored to different levels of understanding.  I'll structure it into sections: **The Core Idea**, **Types of Machine Learning**, **How It Works (Simplified)**, **Examples**, and **Why It's Useful**.\n\n**1. The Core Idea: Learning from Data**\n\nAt its heart, machine learning (ML) is about enabling computers",
                            "reasoning_content": None,
                            "role": "assistant",
                            "tool_calls": [],
                        },
                        "stop_reason": None,
                    }
                ],
                "created": 1759863903,
                "id": "chatcmpl-aaa4288f-2b8e-4bc0-8b14-4e444decd2c4",
                "model": "google/gemma-3-12b-it",
                "object": "chat.completion",
                "prompt_logprobs": None,
                "usage": {
                    "completion_tokens": 100,
                    "prompt_tokens": 14,
                    "prompt_tokens_details": None,
                    "total_tokens": 114,
                },
            },
        }
        
        # Mock the async HTTP handler and Vertex authentication
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler"
        ) as mock_http_handler, patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM._ensure_access_token",
            return_value=("fake-access-token", "PROJECT_ID")
        ):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_vertex_response
            mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)
            
            # Call litellm.acompletion()
            response = await litellm.acompletion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "What is machine learning?"}],
                max_tokens=100,
                api_base="https://32277599999999999.us-central1-10582012152.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
            )
            
            # Verify the request sent to Vertex
            call_args = mock_http_handler.return_value.post.call_args
            assert call_args is not None, "HTTP handler was not called"
            
            request_data = call_args.kwargs["json"]
            print("request body=", json.dumps(request_data, indent=4))
            request_url = call_args.kwargs["url"]
            
            # Validate exact URL matches what we sent
            expected_url = "https://32277599999999999.us-central1-10582012152.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict"
            assert request_url == expected_url, f"Expected URL: {expected_url}\nActual URL: {request_url}"
            
            # Validate Request Body matches expected format
            assert "instances" in request_data
            assert len(request_data["instances"]) == 1
            
            instance = request_data["instances"][0]
            assert instance["@requestFormat"] == "chatCompletions"
            
            # Messages should be directly in the instance, not double-nested
            assert "messages" in instance
            assert instance["messages"][0]["role"] == "user"
            assert instance["messages"][0]["content"] == "What is machine learning?"
            assert instance["max_tokens"] == 100
            
            # Verify stream parameter is NOT sent to Vertex (will be faked client-side)
            assert "stream" not in instance
            
            # Validate LiteLLM Response (OpenAI format)
            assert response.id == "chatcmpl-aaa4288f-2b8e-4bc0-8b14-4e444decd2c4"
            assert response.object == "chat.completion"
            assert response.created == 1759863903
            # Model name has the gemma/ prefix stripped during processing
            assert response.model == "gemma-3-12b-it-1222199011122"
            
            # Validate choices
            assert len(response.choices) == 1
            assert response.choices[0].index == 0
            assert response.choices[0].finish_reason == "length"
            assert response.choices[0].message.role == "assistant"
            assert "machine learning" in response.choices[0].message.content.lower()
            
            # Validate usage
            assert response.usage.prompt_tokens == 14
            assert response.usage.completion_tokens == 100
            assert response.usage.total_tokens == 114

    @pytest.mark.asyncio
    async def test_acompletion_error_handling(self):
        """
        Test litellm.acompletion() error handling when Vertex returns invalid response
        
        Expected: Proper error handling when 'predictions' field is missing
        """
        from litellm.exceptions import APIConnectionError

        # Invalid response without predictions field
        invalid_response = {
            "deployedModelId": "123",
            "error": {
                "code": 400,
                "message": "Invalid request"
            }
        }
        
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler"
        ) as mock_http_handler, patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM._ensure_access_token",
            return_value=("fake-access-token", "test-project")
        ):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = invalid_response
            mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)
            
            # Should raise exception (wrapped as APIConnectionError by LiteLLM)
            with pytest.raises(APIConnectionError) as exc_info:
                await litellm.acompletion(
                    model="vertex_ai/gemma/gemma-3-12b-it",
                    messages=[{"role": "user", "content": "Test"}],
                    api_base="https://test.prediction.vertexai.goog/v1/projects/test/locations/us-central1/endpoints/123:predict",
                    vertex_project="test-project",
                    vertex_location="us-central1",
                )
            
            # Verify the error message contains the original error
            assert "missing 'predictions' field" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_acompletion_fake_streaming(self):
        """
        Test that streaming requests are faked properly for Vertex AI Gemma models.
        
        Verifies:
        1. Request body does NOT include 'stream' parameter (model doesn't support it)
        2. Response returns a MockResponseIterator that yields chunks
        """
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator

        # Mock Vertex response
        mock_vertex_response = {
            "deployedModelId": "1207280419999999999",
            "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
            "modelDisplayName": "gemma-3-12b-it-1222199011122",
            "modelVersionId": "1",
            "predictions": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": "Streaming test response",
                            "reasoning_content": None,
                            "role": "assistant",
                            "tool_calls": [],
                        },
                        "stop_reason": None,
                    }
                ],
                "created": 1759863903,
                "id": "chatcmpl-test-stream",
                "model": "google/gemma-3-12b-it",
                "object": "chat.completion",
                "prompt_logprobs": None,
                "usage": {
                    "completion_tokens": 3,
                    "prompt_tokens": 10,
                    "prompt_tokens_details": None,
                    "total_tokens": 13,
                },
            },
        }
        
        with patch(
            "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM._ensure_access_token",
            return_value=("fake-access-token", "PROJECT_ID")
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_vertex_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client
            
            # Call litellm.acompletion() with stream=True
            response = await litellm.acompletion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "Test streaming"}],
                stream=True,
                api_base="https://test.us-central1-project.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
            )
            
            # Verify the response is a MockResponseIterator
            assert isinstance(response, MockResponseIterator), f"Expected MockResponseIterator, got {type(response)}"
            
            # Verify the request sent to Vertex does NOT include 'stream'
            call_args = mock_client.post.call_args
            assert call_args is not None, "HTTP client was not called"
            
            request_data = call_args.kwargs["json"]
            instance = request_data["instances"][0]
            
            # Critical: Verify stream parameter is NOT sent to Vertex API
            assert "stream" not in instance, "stream parameter should not be sent to Vertex API"
            
            # Verify we can iterate the fake stream and get the response
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
            
            # Should get exactly one chunk (fake streaming)
            assert len(chunks) == 1, f"Expected 1 chunk from fake stream, got {len(chunks)}"
            
            # Verify the chunk has the expected content
            chunk = chunks[0]
            assert hasattr(chunk, "choices")
            assert len(chunk.choices) > 0
            assert chunk.choices[0].delta.content == "Streaming test response"

    @pytest.mark.asyncio
    async def test_acompletion_filters_stream_and_stream_options(self):
        """
        Test that both stream and stream_options are filtered out from the request.
        
        Verifies that when stream=True and stream_options={'include_usage': True} are passed,
        neither parameter is sent to the Vertex API since Vertex Gemma doesn't support them.
        """
        # Mock Vertex response
        mock_vertex_response = {
            "deployedModelId": "1207280419999999999",
            "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
            "modelDisplayName": "gemma-3-12b-it-1222199011122",
            "modelVersionId": "1",
            "predictions": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": "Test response",
                            "reasoning_content": None,
                            "role": "assistant",
                            "tool_calls": [],
                        },
                        "stop_reason": None,
                    }
                ],
                "created": 1759863903,
                "id": "chatcmpl-test",
                "model": "google/gemma-3-12b-it",
                "object": "chat.completion",
                "prompt_logprobs": None,
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 10,
                    "prompt_tokens_details": None,
                    "total_tokens": 12,
                },
            },
        }
        
        with patch(
            "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM._ensure_access_token",
            return_value=("fake-access-token", "PROJECT_ID")
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_vertex_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client
            
            # Call with both stream and stream_options
            response = await litellm.acompletion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "Test"}],
                stream=True,
                stream_options={"include_usage": True},
                api_base="https://test.us-central1-project.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
            )
            
            # Verify the request sent to Vertex
            call_args = mock_client.post.call_args
            assert call_args is not None, "HTTP client was not called"
            
            request_data = call_args.kwargs["json"]
            print("request body=", json.dumps(request_data, indent=4))
            instance = request_data["instances"][0]
            
            # Critical: Verify both stream and stream_options are NOT sent to Vertex API
            assert "stream" not in instance, "stream parameter should not be sent to Vertex API"
            assert "stream_options" not in instance, "stream_options parameter should not be sent to Vertex API"
            
            # Verify other parameters are present
            assert "messages" in instance
            assert instance["@requestFormat"] == "chatCompletions"

