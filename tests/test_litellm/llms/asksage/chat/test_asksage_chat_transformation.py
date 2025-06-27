"""
Unit tests for AskSage AI configuration.

These tests validate the AskSageConfig class which provides AskSage-specific
transformation and configuration for chat completions.
"""

import os
import sys
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.asksage.chat.transformation import AskSageConfig, AskSageException


class TestAskSageConfig:
    """Test class for AskSageConfig functionality"""

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        config = AskSageConfig()
        headers = {}
        api_key = "fake-asksage-key"

        result = config.validate_environment(
            headers=headers,
            model="asksage/default",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://api.asksage.ai/server",
        )

        # Verify headers - AskSage uses x-access-tokens instead of Bearer
        assert result["x-access-tokens"] == api_key
        assert result["Content-Type"] == "application/json"

    def test_missing_api_key(self):
        """Test error handling when API key is missing"""
        config = AskSageConfig()

        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="asksage/default",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://api.asksage.ai/server",
            )

        assert "AskSage API key is required" in str(excinfo.value)

    def test_get_complete_url(self):
        """Test that get_complete_url returns correct endpoint"""
        config = AskSageConfig()
        
        url = config.get_complete_url(
            api_base="https://api.asksage.ai/server",
            api_key="fake-key",
            model="asksage/default",
            optional_params={},
            litellm_params={},
        )
        
        assert url == "https://api.asksage.ai/server/query"

    def test_get_complete_url_default_base(self):
        """Test that get_complete_url uses default API base when none provided"""
        config = AskSageConfig()
        
        url = config.get_complete_url(
            api_base=None,
            api_key="fake-key",
            model="asksage/default",
            optional_params={},
            litellm_params={},
        )
        
        assert url == "https://api.asksage.ai/server/query"

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns correct parameters"""
        config = AskSageConfig()
        
        supported_params = config.get_supported_openai_params("asksage/default")
        
        expected_params = [
            "messages",
            "model", 
            "temperature",
            "persona",
            "system_prompt",
            "dataset",
            "limit_references",
            "live",
        ]
        
        for param in expected_params:
            assert param in supported_params

    def test_map_openai_params(self):
        """Test map_openai_params handles parameters correctly"""
        config = AskSageConfig()

        # Test with supported parameters
        non_default_params = {
            "temperature": 0.7,
            "persona": "helpful",
            "system_prompt": "You are a helpful assistant",
            "dataset": "general",
            "limit_references": 5,
            "live": 1,
            "max_tokens": 100,  # Should be dropped
            "top_p": 0.9,  # Should be dropped
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="asksage/default",
            drop_params=True,
        )
        
        # Check that supported params are mapped
        assert result["temperature"] == 0.7
        assert result["persona"] == "helpful"
        assert result["system_prompt"] == "You are a helpful assistant"
        assert result["dataset"] == "general"
        assert result["limit_references"] == 5
        assert result["live"] == 1
        
        # Check that unsupported params are dropped
        assert "max_tokens" not in non_default_params
        assert "top_p" not in non_default_params

    def test_transform_request_basic(self):
        """Test basic request transformation"""
        config = AskSageConfig()
        
        messages = [
            {"role": "user", "content": "What is the weather like?"}
        ]
        
        result = config.transform_request(
            model="asksage/default",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        
        expected = {
            "message": "What is the weather like?",
            "model": "asksage/default",
            "temperature": 0,
            "limit_references": 0,
            "live": 0,
        }
        
        assert result == expected

    def test_transform_request_with_system_prompt(self):
        """Test request transformation with system prompt from messages"""
        config = AskSageConfig()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the weather like?"}
        ]
        
        result = config.transform_request(
            model="asksage/default",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        
        assert result["message"] == "What is the weather like?"
        assert result["system_prompt"] == "You are a helpful assistant"

    def test_transform_request_with_optional_params(self):
        """Test request transformation with optional parameters"""
        config = AskSageConfig()
        
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        optional_params = {
            "temperature": 0.8,
            "persona": "friendly",
            "dataset": "custom",
            "limit_references": 3,
            "live": 1,
        }
        
        result = config.transform_request(
            model="asksage/default",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        
        assert result["temperature"] == 0.8
        assert result["persona"] == "friendly"
        assert result["dataset"] == "custom"
        assert result["limit_references"] == 3
        assert result["live"] == 1

    def test_transform_request_multipart_content(self):
        """Test request transformation with multipart content"""
        config = AskSageConfig()
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "How are you?"}
                ]
            }
        ]
        
        result = config.transform_request(
            model="asksage/default",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        
        assert result["message"] == "Hello How are you?"

    def test_transform_response_success(self):
        """Test successful response transformation"""
        from unittest.mock import Mock
        from litellm.types.utils import ModelResponse
        
        config = AskSageConfig()
        
        # Mock response
        raw_response = Mock()
        raw_response.json.return_value = {"message": "Hello! How can I help you?"}
        raw_response.status_code = 200
        
        model_response = ModelResponse()
        
        result = config.transform_response(
            model="asksage/default",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=None,
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        
        assert result.model == "asksage/default"
        assert result.object == "chat.completion"
        assert len(result.choices) == 1
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].message.content == "Hello! How can I help you?"
        assert result.choices[0].finish_reason == "stop"

    def test_transform_response_invalid_json(self):
        """Test response transformation with invalid JSON"""
        from unittest.mock import Mock
        from litellm.types.utils import ModelResponse
        import json
        
        config = AskSageConfig()
        
        # Mock response with invalid JSON
        raw_response = Mock()
        raw_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        raw_response.text = "Invalid response"
        raw_response.status_code = 500
        raw_response.headers = {}
        
        model_response = ModelResponse()
        
        with pytest.raises(AskSageException) as excinfo:
            config.transform_response(
                model="asksage/default",
                raw_response=raw_response,
                model_response=model_response,
                logging_obj=None,
                request_data={},
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                encoding=None,
            )
        
        assert "Invalid JSON response" in str(excinfo.value)
        assert excinfo.value.status_code == 500

    def test_get_error_class(self):
        """Test that get_error_class returns AskSageException"""
        config = AskSageConfig()
        
        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"}
        )
        
        assert isinstance(error, AskSageException)
        assert error.status_code == 400
        assert "Test error" in str(error)

    def test_asksage_completion_mock(self, respx_mock):
        """
        Mock test for AskSage completion.
        This test mocks the actual HTTP request to test the integration properly.
        """
        import litellm

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )
        from litellm import completion

        # Set up environment variables for the test
        api_key = "fake-asksage-key"
        api_base = "https://api.asksage.ai/server"
        model = "asksage/default"

        # Mock the HTTP request to the AskSage API
        respx_mock.post(f"{api_base}/query").respond(
            json={
                "message": "Hello! I'm an AI assistant created by AskSage. How can I help you today?"
            },
            status_code=200,
        )

        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "Hello, who are you?"}
            ],
            api_key=api_key,
            api_base=api_base,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert response.choices[0].message.content is not None

        # Check for specific content in the response
        assert "AskSage" in response.choices[0].message.content
        assert "AI assistant" in response.choices[0].message.content

    def test_custom_llm_provider_property(self):
        """Test that custom_llm_provider returns correct value"""
        config = AskSageConfig()
        assert config.custom_llm_provider == "asksage"

    def test_developer_role_translation(self):
        """Test that developer role gets translated to system role"""
        config = AskSageConfig()
        
        messages = [
            {"role": "developer", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        
        result = config.transform_request(
            model="asksage/default",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        
        # The developer message should be converted to system_prompt
        assert result["system_prompt"] == "You are a helpful assistant"
        assert result["message"] == "Hello"
