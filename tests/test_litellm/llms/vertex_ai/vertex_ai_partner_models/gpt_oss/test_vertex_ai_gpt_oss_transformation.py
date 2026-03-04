import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.vertex_ai_partner_models.gpt_oss.transformation import (
    VertexAIGPTOSSTransformation,
)


@pytest.fixture(autouse=True)
def _reset_litellm_http_client_cache():
    """Ensure each test gets a fresh async HTTP client mock."""
    from litellm import in_memory_llm_clients_cache

    in_memory_llm_clients_cache.flush_cache()


@pytest.fixture(autouse=True)
def clean_vertex_env():
    """Clear Google/Vertex AI environment variables before each test to prevent test isolation issues."""
    saved_env = {}
    env_vars_to_clear = [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "VERTEXAI_CREDENTIALS",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "VERTEX_AI_PROJECT",
    ]
    for var in env_vars_to_clear:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    yield

    # Restore saved environment variables
    for var, value in saved_env.items():
        os.environ[var] = value


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

    mock_vertexai = MagicMock()
    mock_vertexai.preview = MagicMock()
    mock_vertexai.preview.language_models = MagicMock()

    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler") as mock_http_handler, \
         patch("litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM._ensure_access_token",
               return_value=("fake-token", "pathrise-convert-1606954137718")), \
         patch.dict("sys.modules", {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview}), \
         patch.dict(os.environ, {"VERTEXAI_PROJECT": "pathrise-convert-1606954137718"}):
        mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)

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
        )

        # Verify the mock was called
        mock_http_handler.return_value.post.assert_called_once()

        # Get the call arguments
        call_args = mock_http_handler.return_value.post.call_args
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

    mock_vertexai = MagicMock()
    mock_vertexai.preview = MagicMock()
    mock_vertexai.preview.language_models = MagicMock()

    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler") as mock_http_handler, \
         patch("litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM._ensure_access_token",
               return_value=("fake-token", "pathrise-convert-1606954137718")), \
         patch.dict("sys.modules", {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview}), \
         patch.dict(os.environ, {"VERTEXAI_PROJECT": "pathrise-convert-1606954137718"}):
        mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)

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
        )

        # Verify the mock was called
        mock_http_handler.return_value.post.assert_called_once()

        # Get the call arguments
        call_args = mock_http_handler.return_value.post.call_args
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
