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

import httpx
import pytest

import litellm
from litellm import completion


class TestDockerModelRunnerIntegration:
    """Integration test for Docker Model Runner"""

    def test_completion_hits_correct_url_and_body(self):
        """
        Test that litellm.completion with docker_model_runner provider:
        1. Hits the correct URL: {api_base}/v1/chat/completions where api_base includes engine path
        2. Sends the correct request body with messages and parameters
        """
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
            # Mock the response
            mock_response = Mock(spec=httpx.Response)
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
            mock_response.headers = httpx.Headers({"content-type": "application/json"})
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

    def test_completion_with_custom_engine_and_host(self):
        """
        Test that litellm.completion works with custom engine and host:
        1. Uses model-runner.docker.internal as host
        2. Specifies a different engine in the api_base
        3. Model name is sent in the request body
        """
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
            # Mock the response
            mock_response = Mock(spec=httpx.Response)
            mock_response.json.return_value = {
                "id": "chatcmpl-456",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "mistral-7b",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Bonjour! How can I assist you?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40
                }
            }
            mock_response.status_code = 200
            mock_response.headers = httpx.Headers({"content-type": "application/json"})
            mock_post.return_value = mock_response

            # Make the completion call with custom engine and host
            response = completion(
                model="docker_model_runner/mistral-7b",
                messages=[{"role": "user", "content": "Hello!"}],
                api_base="http://model-runner.docker.internal/engines/custom-engine",
                temperature=0.5,
                max_tokens=200
            )

            # Verify the URL was correct
            assert mock_post.called
            call_args = mock_post.call_args
            url = call_args[1]["url"]
            print("URL For request", url)
            print("request body for request", json.dumps(call_args[1]["data"], indent=4))
            
            # Should hit the custom host and engine
            assert "model-runner.docker.internal" in url
            assert "/engines/custom-engine/v1/chat/completions" in url

            # Verify the request body contains the model name
            request_data = call_args[1]["data"]
            if isinstance(request_data, str):
                request_data = json.loads(request_data)
            
            # Check that model name is in the request body
            assert request_data["model"] == "mistral-7b"
            
            # Check messages
            assert "messages" in request_data
            assert len(request_data["messages"]) == 1
            assert request_data["messages"][0]["role"] == "user"
            assert request_data["messages"][0]["content"] == "Hello!"
            
            # Check parameters
            assert request_data["temperature"] == 0.5
            assert request_data["max_tokens"] == 200
            
            # Verify response
            assert response.choices[0].message.content == "Bonjour! How can I assist you?"

