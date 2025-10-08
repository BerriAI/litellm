import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.vertex_ai_partner_models.gpt_oss.transformation import (
    VertexAIGPTOSSTransformation,
)


class TestVertexAIGPTOSSTransformation:
    """Test class for VertexAI GPT-OSS transformation functionality."""

    def test_supports_reasoning_effort(self):
        """Test that reasoning_effort parameter is supported for GPT-OSS models."""
        config = VertexAIGPTOSSTransformation()
        supported_params = config.get_supported_openai_params(model="openai/gpt-oss-20b-maas")
        
        assert "reasoning_effort" in supported_params

    def test_removes_tool_calling_params_when_not_supported(self):
        """Test that tool calling parameters are removed when function calling is not supported."""
        config = VertexAIGPTOSSTransformation()
        
        # Mock litellm.supports_function_calling to return False
        with patch('litellm.supports_function_calling', return_value=False):
            supported_params = config.get_supported_openai_params(model="openai/gpt-oss-20b-maas")
            
            # Tool calling params should be removed
            assert "tool" not in supported_params
            assert "tool_choice" not in supported_params
            assert "function_call" not in supported_params
            assert "functions" not in supported_params
            
            # But reasoning_effort should still be there
            assert "reasoning_effort" in supported_params


@pytest.mark.asyncio
async def test_vertex_ai_gpt_oss_simple_request():
    """
    Test that a simple request to vertex_ai/openai/gpt-oss-20b-maas lands at the correct URL 
    with the correct request body.
    """
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexLLM,
    )

    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "openai/gpt-oss-20b-maas",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm Litellm Bot, a helpful assistant. I don't have access to real-time weather information, but I'd be happy to help you with other questions or tasks!"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 28,
            "total_tokens": 70
        }
    }
    
    client = AsyncHTTPHandler()
    
    async def mock_post_func(*args, **kwargs):
        return mock_response
    
    with patch.object(client, "post", side_effect=mock_post_func) as mock_post, \
         patch.object(VertexLLM, "_ensure_access_token", return_value=("fake-token", "pathrise-convert-1606954137718")):
        response = await litellm.acompletion(
            model="vertex_ai/openai/gpt-oss-20b-maas",
            messages=[
                {
                    "role": "system",
                    "content": "Your name is Litellm Bot, you are a helpful assistant"
                },
                {
                    "role": "user", 
                    "content": "Hello, what is your name and can you tell me the weather?"
                }
            ],
            vertex_ai_location="us-central1",
            vertex_ai_project="pathrise-convert-1606954137718",
            client=client
        )
        
        # Verify the mock was called
        mock_post.assert_called_once()
        
        # Get the call arguments
        call_args = mock_post.call_args
        # For side_effect, the URL is passed as kwargs['url']
        called_url = call_args.kwargs["url"]
        request_body = json.loads(call_args.kwargs["data"])
        
        # Verify the URL
        expected_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/pathrise-convert-1606954137718/locations/us-central1/endpoints/openapi/chat/completions"
        assert called_url == expected_url
        
        # Verify the request body
        expected_request_body = {
            'model': 'openai/gpt-oss-20b-maas',
            'messages': [
                {
                    'role': 'system',
                    'content': 'Your name is Litellm Bot, you are a helpful assistant'
                },
                {
                    'role': 'user',
                    'content': 'Hello, what is your name and can you tell me the weather?'
                }
            ],
            'stream': False
        }
        assert request_body == expected_request_body
        
        # Verify response structure
        assert response.model == "openai/gpt-oss-20b-maas"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"


@pytest.mark.asyncio
async def test_vertex_ai_gpt_oss_reasoning_effort():
    """
    Test that reasoning_effort parameter is correctly passed in the request body 
    for GPT-OSS models.
    """
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexLLM,
    )

    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = {
        "id": "chatcmpl-test456",
        "object": "chat.completion", 
        "created": 1234567890,
        "model": "openai/gpt-oss-20b-maas",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I need to think about this carefully. The weather varies by location and time, so I would need to know your specific location to provide accurate weather information."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 35,
            "completion_tokens": 32,
            "total_tokens": 67
        }
    }
    
    client = AsyncHTTPHandler()
    
    async def mock_post_func(*args, **kwargs):
        return mock_response
    
    with patch.object(client, "post", side_effect=mock_post_func) as mock_post, \
         patch.object(VertexLLM, "_ensure_access_token", return_value=("fake-token", "pathrise-convert-1606954137718")):
        response = await litellm.acompletion(
            model="vertex_ai/openai/gpt-oss-20b-maas",
            messages=[
                {
                    "role": "system",
                    "content": "Your name is Litellm Bot, you are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": "Hello, what is your name and can you tell me the weather?"
                }
            ],
            reasoning_effort="low",
            vertex_ai_location="us-central1",
            vertex_ai_project="pathrise-convert-1606954137718",
            client=client
        )
        
        # Verify the mock was called
        mock_post.assert_called_once()
        
        # Get the call arguments  
        call_args = mock_post.call_args
        request_body = json.loads(call_args.kwargs["data"])
        
        # Verify reasoning_effort is in the request body
        assert "reasoning_effort" in request_body
        assert request_body["reasoning_effort"] == "low"
        
        # Verify other expected fields
        expected_request_body = {
            'model': 'openai/gpt-oss-20b-maas',
            'messages': [
                {
                    'role': 'system',
                    'content': 'Your name is Litellm Bot, you are a helpful assistant'
                },
                {
                    'role': 'user',
                    'content': 'Hello, what is your name and can you tell me the weather?'
                }
            ],
            'reasoning_effort': 'low',
            'stream': False
        }
        assert request_body == expected_request_body
        
        # Verify response structure
        assert response.model == "openai/gpt-oss-20b-maas"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
