"""
Unit tests for Docker Model Runner transformation.

This test validates that the DockerModelRunnerChatConfig correctly transforms
requests to the proper URL, headers, and body format.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)

import json
from typing import cast

import pytest

from litellm.llms.docker_model_runner.chat.transformation import (
    DockerModelRunnerChatConfig,
)
from litellm.types.llms.openai import AllMessageValues


class TestDockerModelRunnerTransformation:
    """
    Unit tests for Docker Model Runner transformation layer.
    """

    def test_get_complete_url_with_default_api_base(self):
        """
        Test that get_complete_url returns the correct URL with default api_base.
        """
        config = DockerModelRunnerChatConfig()
        
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="llama-3.1",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        assert url == "http://localhost:22088/engines/llama.cpp/v1/chat/completions"

    def test_get_complete_url_with_custom_api_base(self):
        """
        Test that get_complete_url correctly appends /v1/chat/completions to custom api_base.
        """
        config = DockerModelRunnerChatConfig()
        
        url = config.get_complete_url(
            api_base="http://localhost:22088/engines/llama.cpp",
            api_key=None,
            model="llama-3.1",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        assert url == "http://localhost:22088/engines/llama.cpp/v1/chat/completions"
        assert "/engines/llama.cpp/v1/chat/completions" in url
        assert "http://localhost:22088" in url

    def test_get_complete_url_with_custom_engine_and_host(self):
        """
        Test that get_complete_url works with custom engine and host.
        """
        config = DockerModelRunnerChatConfig()
        
        url = config.get_complete_url(
            api_base="http://model-runner.docker.internal/engines/custom-engine",
            api_key=None,
            model="mistral-7b",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        assert "model-runner.docker.internal" in url
        assert "/engines/custom-engine/v1/chat/completions" in url
        assert url == "http://model-runner.docker.internal/engines/custom-engine/v1/chat/completions"

    def test_get_complete_url_removes_trailing_slash(self):
        """
        Test that get_complete_url removes trailing slashes from api_base.
        """
        config = DockerModelRunnerChatConfig()
        
        url = config.get_complete_url(
            api_base="http://localhost:22088/engines/llama.cpp/",
            api_key=None,
            model="llama-3.1",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        # Should not have double slashes
        assert "/v1/chat/completions" in url
        assert "//v1" not in url

    def test_transform_request_body(self):
        """
        Test that transform_request creates the correct request body with messages and parameters.
        """
        config = DockerModelRunnerChatConfig()
        
        messages = cast(list[AllMessageValues], [{"role": "user", "content": "Hello, how are you?"}])
        optional_params = {
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        request_data = config.transform_request(
            model="llama-3.1",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Check messages
        assert "messages" in request_data
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["messages"][0]["content"] == "Hello, how are you?"
        
        # Check parameters
        assert request_data["temperature"] == 0.7
        assert request_data["max_tokens"] == 100
        
        # Check model name is in request
        assert request_data["model"] == "llama-3.1"

    def test_validate_environment_returns_headers(self):
        """
        Test that validate_environment returns the correct headers.
        """
        config = DockerModelRunnerChatConfig()
        
        headers = config.validate_environment(
            headers={},
            model="llama-3.1",
            messages=cast(list[AllMessageValues], [{"role": "user", "content": "Hello"}]),
            optional_params={},
            litellm_params={},
            api_key="test-key",
            api_base="http://localhost:22088/engines/llama.cpp"
        )
        
        # Should have Authorization header with Bearer token
        assert "Authorization" in headers
        assert "Bearer" in headers["Authorization"]

    def test_map_openai_params(self):
        """
        Test that map_openai_params correctly maps OpenAI parameters.
        """
        config = DockerModelRunnerChatConfig()
        
        non_default_params = {
            "temperature": 0.5,
            "max_tokens": 200,
            "top_p": 0.9
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="mistral-7b",
            drop_params=False
        )
        
        # Check that parameters are mapped correctly
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9

    def test_map_max_completion_tokens_to_max_tokens(self):
        """
        Test that max_completion_tokens is mapped to max_tokens.
        """
        config = DockerModelRunnerChatConfig()
        
        non_default_params = {
            "max_completion_tokens": 150
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="llama-3.1",
            drop_params=False
        )
        
        # max_completion_tokens should be mapped to max_tokens
        assert result["max_tokens"] == 150
        assert "max_completion_tokens" not in result

