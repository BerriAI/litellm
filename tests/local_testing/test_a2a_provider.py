"""
Tests for the A2A (Agent-to-Agent) Provider

This test file verifies the A2A provider integration with LiteLLM.

To run these tests:
1. Start an A2A agent (e.g., the helloworld sample agent on port 9999)
2. Run: pytest tests/local_testing/test_a2a_provider.py -v

For the helloworld agent:
    cd a2a-samples/samples/python/agents/helloworld
    python __main__.py

Environment variables:
    A2A_AGENT_API_BASE: Base URL for the A2A agent (default: http://localhost:9999)
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import httpx

import litellm
from litellm.llms.a2a.chat.transformation import A2AAgentConfig, A2AAgentError


# Test fixtures
@pytest.fixture
def a2a_config():
    return A2AAgentConfig()


@pytest.fixture
def sample_openai_messages():
    return [
        {"role": "user", "content": "Hello!"}
    ]


@pytest.fixture
def sample_a2a_response():
    return {
        "jsonrpc": "2.0",
        "id": "test-123",
        "result": {
            "kind": "message",
            "messageId": "msg-123",
            "parts": [
                {"kind": "text", "text": "Hello World"}
            ],
            "role": "agent"
        }
    }


class TestA2AAgentConfig:
    """Test the A2AAgentConfig class."""

    def test_custom_llm_provider(self, a2a_config):
        """Test that the custom_llm_provider is correctly set."""
        assert a2a_config.custom_llm_provider == "a2a_agent"

    def test_get_supported_openai_params(self, a2a_config):
        """Test that supported params are returned."""
        params = a2a_config.get_supported_openai_params("test-model")
        assert "stream" in params
        assert "max_tokens" in params
        assert "temperature" in params

    def test_validate_environment_missing_api_base(self, a2a_config):
        """Test that validation fails when api_base is missing."""
        with pytest.raises(A2AAgentError):
            a2a_config.validate_environment(
                headers={},
                model="test-model",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

    def test_validate_environment_with_api_base(self, a2a_config):
        """Test that validation succeeds with api_base."""
        headers = a2a_config.validate_environment(
            headers={},
            model="test-model",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
            api_base="http://localhost:9999",
        )
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" in headers


class TestA2AMessageTransformation:
    """Test OpenAI to A2A message transformation."""

    def test_transform_simple_message(self, a2a_config, sample_openai_messages):
        """Test transforming a simple user message."""
        request = a2a_config.transform_request(
            model="test-model",
            messages=sample_openai_messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        
        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "message/send"
        assert "params" in request
        assert "message" in request["params"]
        
        message = request["params"]["message"]
        assert message["role"] == "user"
        assert len(message["parts"]) == 1
        assert message["parts"][0]["kind"] == "text"
        assert message["parts"][0]["text"] == "Hello!"

    def test_transform_streaming_message(self, a2a_config, sample_openai_messages):
        """Test transforming a message with streaming enabled."""
        request = a2a_config.transform_request(
            model="test-model",
            messages=sample_openai_messages,
            optional_params={"stream": True},
            litellm_params={},
            headers={},
        )
        
        assert request["method"] == "message/stream"

    def test_transform_multipart_content(self, a2a_config):
        """Test transforming messages with multipart content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
                ]
            }
        ]
        
        request = a2a_config.transform_request(
            model="test-model",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        
        message = request["params"]["message"]
        assert len(message["parts"]) == 2
        assert message["parts"][0]["kind"] == "text"
        assert message["parts"][1]["kind"] == "file"


class TestA2AResponseTransformation:
    """Test A2A to OpenAI response transformation."""

    def test_extract_message_content(self, a2a_config, sample_a2a_response):
        """Test extracting content from A2A message response."""
        result = sample_a2a_response["result"]
        content = a2a_config._extract_content_from_a2a_result(result)
        assert content == "Hello World"

    def test_extract_task_artifact_content(self, a2a_config):
        """Test extracting content from A2A task with artifacts."""
        result = {
            "kind": "task",
            "id": "task-123",
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "artifactId": "artifact-1",
                    "parts": [{"kind": "text", "text": "Task result"}]
                }
            ]
        }
        content = a2a_config._extract_content_from_a2a_result(result)
        assert content == "Task result"

    def test_determine_finish_reason_completed(self, a2a_config):
        """Test determining finish reason for completed task."""
        result = {
            "task": {
                "status": {"state": "completed"}
            }
        }
        reason = a2a_config._determine_finish_reason(result)
        assert reason == "stop"

    def test_determine_finish_reason_input_required(self, a2a_config):
        """Test determining finish reason for input_required task."""
        result = {
            "task": {
                "status": {"state": "input_required"}
            }
        }
        reason = a2a_config._determine_finish_reason(result)
        assert reason == "stop"


class TestA2AIntegration:
    """Integration tests requiring a running A2A agent."""

    @pytest.mark.skip(reason="Requires running A2A agent")
    def test_non_streaming_completion(self):
        """Test non-streaming completion with real A2A agent."""
        api_base = os.environ.get("A2A_AGENT_API_BASE", "http://localhost:9999")
        
        response = litellm.completion(
            model="a2a_agent/test-agent",
            messages=[{"role": "user", "content": "Hello!"}],
            api_base=api_base,
        )
        
        assert response.id is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.skip(reason="Requires running A2A agent")
    def test_streaming_completion(self):
        """Test streaming completion with real A2A agent."""
        api_base = os.environ.get("A2A_AGENT_API_BASE", "http://localhost:9999")
        
        response = litellm.completion(
            model="a2a_agent/test-agent",
            messages=[{"role": "user", "content": "Hello!"}],
            api_base=api_base,
            stream=True,
        )
        
        content = ""
        finish_reason = None
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        
        assert content != ""
        assert finish_reason == "stop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
