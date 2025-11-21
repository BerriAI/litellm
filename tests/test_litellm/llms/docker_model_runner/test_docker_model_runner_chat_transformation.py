"""
Unit tests for Docker Model Runner configuration.

This test validates that litellm.completion correctly routes requests to Docker Model Runner
with the proper URL structure and request body.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)

import json
from unittest.mock import Mock, patch

import pytest

import litellm
from litellm import completion


class TestDockerModelRunnerIntegration:
    """Integration test for Docker Model Runner"""

    @pytest.mark.asyncio
    async def test_completion_hits_correct_url_and_body(self):
        """
        Test that litellm.completion with docker_model_runner provider:
        1. Hits the correct URL: {api_base}/v1/chat/completions where api_base includes engine path
        2. Sends the correct request body with messages and parameters
        """
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
            # Mock the response
            mock_response = Mock()
            mock_response.json.return_value = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "llama-3.1",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_post.return_value = mock_response

            # Make the completion call with engine in api_base
            response = completion(
                model="docker_model_runner/llama-3.1",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                api_base="http://localhost:22088/engines/llama.cpp",
                temperature=0.7,
                max_tokens=100
            )

            # Verify the URL was correct
            assert mock_post.called
            call_args = mock_post.call_args
            url = call_args[1]["url"]
            print("URL For request", url)
            print("request body for request", json.dumps(call_args[1]["data"], indent=4))
            
            # Should hit {api_base}/v1/chat/completions where api_base includes engine
            assert "/engines/llama.cpp/v1/chat/completions" in url
            assert "http://localhost:22088" in url

            # Verify the request body
            request_data = call_args[1]["data"]
            if isinstance(request_data, str):
                request_data = json.loads(request_data)
            
            # Check messages
            assert "messages" in request_data
            assert len(request_data["messages"]) == 1
            assert request_data["messages"][0]["role"] == "user"
            assert request_data["messages"][0]["content"] == "Hello, how are you?"
            
            # Check parameters
            assert request_data["temperature"] == 0.7
            assert request_data["max_tokens"] == 100
            
            # Verify response
            assert response.choices[0].message.content == "Hello! How can I help you today?"

