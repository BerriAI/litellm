"""
Tests for IBM watsonx.ai Orchestrate Agent transformation.
"""

import json
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from httpx import Response

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.watsonx.agents.transformation import IBMWatsonXAgentConfig
from litellm.types.utils import Message, ModelResponse


class TestIBMWatsonXAgentConfig:
    """Test suite for IBMWatsonXAgentConfig methods"""

    @pytest.fixture
    def config(self):
        """Create a test instance of IBMWatsonXAgentConfig"""
        return IBMWatsonXAgentConfig()

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing"""
        return [
            {"role": "user", "content": "Hello, how can you help me?"},
            {"role": "assistant", "content": "I can help with various tasks."},
            {"role": "user", "content": "What is the weather like?"},
        ]

    @pytest.fixture
    def sample_agent_response(self):
        """Sample agent API response"""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "watsonx_agent/abc123",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The weather is sunny today.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "thread_id": "thread-xyz789",
        }

    def test_get_supported_openai_params(self, config):
        """Test getting supported OpenAI parameters"""
        model = "watsonx_agent/test123"
        params = config.get_supported_openai_params(model)

        assert "stream" in params
        assert "messages" in params

    def test_get_agent_id_valid(self, config):
        """Test extracting valid agent_id from model string"""
        model = "watsonx_agent/abc123"
        agent_id = config._get_agent_id(model)

        assert agent_id == "abc123"

    def test_get_agent_id_complex(self, config):
        """Test extracting agent_id with complex format"""
        model = "watsonx_agent/agent-id-with-dashes"
        agent_id = config._get_agent_id(model)

        assert agent_id == "agent-id-with-dashes"

    def test_get_agent_id_invalid_format(self, config):
        """Test extracting agent_id with invalid format raises error"""
        invalid_models = [
            "invalid_model",
            "watsonx_agent",
            "",
        ]

        for invalid_model in invalid_models:
            with pytest.raises(ValueError, match="Invalid model format"):
                config._get_agent_id(invalid_model)

    @patch.object(IBMWatsonXAgentConfig, "_get_base_url")
    def test_get_complete_url(self, mock_get_base_url, config):
        """Test building complete URL"""
        mock_get_base_url.return_value = "https://api.example.com"

        model = "watsonx_agent/abc123"
        url = config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model=model,
            optional_params={},
            litellm_params={},
            stream=False,
        )

        expected_url = "https://api.example.com/api/v1/orchestrate/abc123/chat/completions"
        assert url == expected_url

    @patch.object(IBMWatsonXAgentConfig, "_get_base_url")
    def test_get_complete_url_with_trailing_slash(self, mock_get_base_url, config):
        """Test building complete URL with trailing slash in base URL"""
        mock_get_base_url.return_value = "https://api.example.com/"

        model = "watsonx_agent/abc123"
        url = config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model=model,
            optional_params={},
            litellm_params={},
            stream=False,
        )

        expected_url = "https://api.example.com/api/v1/orchestrate/abc123/chat/completions"
        assert url == expected_url

    def test_transform_messages(self, config, sample_messages):
        """Test transforming OpenAI messages to agent format"""
        transformed = config._transform_messages(sample_messages)

        assert len(transformed) == 3
        assert transformed[0]["role"] == "user"
        assert transformed[0]["content"] == "Hello, how can you help me?"
        assert transformed[2]["role"] == "user"
        assert transformed[2]["content"] == "What is the weather like?"

    def test_transform_messages_with_list_content(self, config):
        """Test transforming messages with list content"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world"},
                ],
            }
        ]

        with patch(
            "litellm.llms.watsonx.agents.transformation.convert_content_list_to_str"
        ) as mock_convert:
            mock_convert.return_value = "Hello world"
            transformed = config._transform_messages(messages)

            assert len(transformed) == 1
            assert transformed[0]["content"] == "Hello world"

    def test_transform_request(self, config, sample_messages):
        """Test transforming complete request"""
        model = "watsonx_agent/abc123"
        optional_params = {
            "stream": True,
            "additional_parameters": {"key": "value"},
            "context": {"user_id": "123"},
        }

        request_data = config.transform_request(
            model=model,
            messages=sample_messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert "messages" in request_data
        assert len(request_data["messages"]) == 3
        assert request_data["stream"] is True
        assert request_data["additional_parameters"] == {"key": "value"}
        assert request_data["context"] == {"user_id": "123"}

    def test_transform_request_defaults(self, config, sample_messages):
        """Test transforming request with default values"""
        model = "watsonx_agent/abc123"
        optional_params = {}

        request_data = config.transform_request(
            model=model,
            messages=sample_messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert "messages" in request_data
        assert request_data["stream"] is True
        assert request_data["additional_parameters"] == {}
        assert request_data["context"] == {}

    def test_transform_agent_choice_to_litellm(self, config):
        """Test transforming agent choice to LiteLLM format"""
        agent_choice = {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }

        litellm_choice = config._transform_agent_choice_to_litellm(agent_choice, 0)

        assert litellm_choice.index == 0
        assert litellm_choice.message.role == "assistant"
        assert litellm_choice.message.content == "Hello!"
        assert litellm_choice.finish_reason == "stop"

    def test_transform_agent_choice_with_dict_content(self, config):
        """Test transforming agent choice with dict content"""
        agent_choice = {
            "index": 0,
            "message": {"role": "assistant", "content": {"key": "value"}},
            "finish_reason": "stop",
        }

        litellm_choice = config._transform_agent_choice_to_litellm(agent_choice, 0)

        assert litellm_choice.message.content == json.dumps({"key": "value"})

    def test_transform_agent_choice_with_list_content(self, config):
        """Test transforming agent choice with list content"""
        agent_choice = {
            "index": 0,
            "message": {"role": "assistant", "content": ["item1", "item2"]},
            "finish_reason": "stop",
        }

        litellm_choice = config._transform_agent_choice_to_litellm(agent_choice, 0)

        assert litellm_choice.message.content == json.dumps(["item1", "item2"])

    def test_transform_agent_choice_missing_finish_reason(self, config):
        """Test transforming agent choice without finish_reason"""
        agent_choice = {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
        }

        litellm_choice = config._transform_agent_choice_to_litellm(agent_choice, 0)

        assert litellm_choice.finish_reason == "stop"

    def test_transform_response(self, config, sample_agent_response):
        """Test transforming complete response"""
        # Create mock response
        mock_response = Mock(spec=Response)
        mock_response.json.return_value = sample_agent_response
        mock_response.status_code = 200

        model_response = ModelResponse()

        result = config.transform_response(
            model="watsonx_agent/abc123",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
        )

        assert result.id == "chatcmpl-123"
        assert result.created == 1677652288
        assert result.model == "watsonx_agent/abc123"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "The weather is sunny today."
        assert result._hidden_params is not None
        assert result._hidden_params.get("thread_id") == "thread-xyz789"

    def test_transform_response_multiple_choices(self, config):
        """Test transforming response with multiple choices"""
        agent_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "watsonx_agent/abc123",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response 1"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Response 2"},
                    "finish_reason": "stop",
                },
            ],
            "thread_id": "thread-xyz789",
        }

        mock_response = Mock(spec=Response)
        mock_response.json.return_value = agent_response
        mock_response.status_code = 200

        model_response = ModelResponse()

        result = config.transform_response(
            model="watsonx_agent/abc123",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
        )

        assert len(result.choices) == 2
        assert result.choices[0].message.content == "Response 1"
        assert result.choices[1].message.content == "Response 2"

    def test_map_openai_params(self, config):
        """Test mapping OpenAI parameters"""
        non_default_params = {
            "thread_id": "thread-123",
            "additional_parameters": {"key": "value"},
            "context": {"user": "test"},
            "unsupported_param": "should_be_ignored",
        }
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="watsonx_agent/abc123",
            drop_params=False,
        )

        assert result["thread_id"] == "thread-123"
        assert result["additional_parameters"] == {"key": "value"}
        assert result["context"] == {"user": "test"}
        # unsupported_param should remain in non_default_params
        assert "unsupported_param" in non_default_params

    @patch.object(IBMWatsonXAgentConfig, "validate_environment")
    def test_validate_environment_called(self, mock_validate, config):
        """Test that validate_environment uses parent class method"""
        mock_validate.return_value = {"Authorization": "Bearer token"}

        headers = config.validate_environment(
            headers={},
            model="watsonx_agent/abc123",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
            api_base="https://api.example.com",
        )

        mock_validate.assert_called_once()

    def test_get_error_class(self, config):
        """Test getting error class"""
        from litellm.llms.watsonx.common_utils import WatsonXAIError

        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={},
        )

        assert isinstance(error, WatsonXAIError)
        assert error.status_code == 400
        assert "Test error" in error.message
