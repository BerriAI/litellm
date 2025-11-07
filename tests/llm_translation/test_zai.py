import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch
from typing import Optional

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
# from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse, Usage
from litellm import completion
from unittest.mock import patch
from litellm.llms.zai.chat.transformation import ZaiChatConfig
from base_llm_unit_tests import BaseLLMChatTest


class TestZaiChatConfig:
    """Test suite for ZaiChatConfig class"""

    def test_validate_environment_with_api_key(self):
        """Test validate_environment method with provided API key"""
        config = ZaiChatConfig()
        headers = {}
        api_key = "test_zai_key"

        result = config.validate_environment(headers, "zai/glm-4.6", [], {}, api_key)

        assert "Authorization" in result
        assert result["Authorization"] == "Bearer test_zai_key"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_missing_key(self):
        """Test validate_environment method when API key is missing"""
        config = ZaiChatConfig()
        headers = {}
        # Ensure the environment variable is not set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing ZAI_API_KEY"):
                config.validate_environment(headers, "zai/glm-4.6", [], {})

    def test_validate_environment_with_env_var(self):
        """Test validate_environment method with environment variable"""
        config = ZaiChatConfig()
        headers = {}
        
        with patch.dict(os.environ, {"ZAI_API_KEY": "env_api_key"}):
            result = config.validate_environment(headers, "zai/glm-4.6", [], {})
            
            assert "Authorization" in result
            assert result["Authorization"] == "Bearer env_api_key"
            assert result["Content-Type"] == "application/json"

    def test_get_complete_url_default(self):
        """Test get_complete_url method with default API base"""
        config = ZaiChatConfig()
        
        result = config.get_complete_url(None, "zai/glm-4.6")
        
        assert result == "https://api.z.ai/api/paas/v4/chat/completions"

    def test_get_complete_url_custom_base(self):
        """Test get_complete_url method with custom API base"""
        config = ZaiChatConfig()
        
        result = config.get_complete_url("https://custom.api.z.ai", "zai/glm-4.6")
        
        assert result == "https://custom.api.z.ai/chat/completions"

    def test_transform_request(self):
        """Test transform_request method with standard OpenAI-style parameters"""
        config = ZaiChatConfig()
        model = "zai/glm-4.6"
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        optional_params = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False,
            "stop": ["END"],
        }

        result = config.transform_request(model, messages, optional_params, {})

        assert result["model"] == "glm-4.6"  # zai/ prefix should be removed
        assert result["messages"] == messages
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stream"] is False
        assert result["stop"] == ["END"]

    def test_transform_request_minimal_params(self):
        """Test transform_request method with minimal parameters"""
        config = ZaiChatConfig()
        model = "zai/glm-4.6"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}

        result = config.transform_request(model, messages, optional_params, {})

        assert result["model"] == "glm-4.6"
        assert result["messages"] == messages
        assert len(result) == 2  # Only model and messages should be present

    def test_transform_response(self):
        """Test transform_response method with sample Zai response"""
        config = ZaiChatConfig()
        model = "zai/glm-4.6"
        raw_response = {
            "id": "chatcmpl-12345",
            "model": "glm-4.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I am doing well, thank you for asking."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
        model_response = litellm.ModelResponse()

        result = config.transform_response(
            model, raw_response, model_response, None, {}, [], {}, {}
        )

        assert result.id == "chatcmpl-12345"
        assert result.model == "glm-4.6"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "I am doing well, thank you for asking."
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 15
        assert result.usage.total_tokens == 25

    def test_transform_response_minimal(self):
        """Test transform_response method with minimal response"""
        config = ZaiChatConfig()
        model = "zai/glm-4.6"
        raw_response = {
            "choices": [
                {
                    "message": {
                        "content": "Hello!"
                    }
                }
            ]
        }
        model_response = litellm.ModelResponse()

        result = config.transform_response(
            model, raw_response, model_response, None, {}, [], {}, {}
        )

        assert result.choices[0].message.content == "Hello!"
        assert result.model == raw_response.get("model")  # Should handle missing model

    def test_get_error_class_authentication_error(self):
        """Test get_error_class for authentication error"""
        config = ZaiChatConfig()
        
        result = config.get_error_class("Invalid API key", 401, {})
        
        assert isinstance(result, litellm.AuthenticationError)
        assert result.llm_provider == "zai"

    def test_get_error_class_bad_request_error(self):
        """Test get_error_class for bad request error"""
        config = ZaiChatConfig()
        
        result = config.get_error_class("Request malformed", 400, {})
        
        assert isinstance(result, litellm.BadRequestError)
        assert result.llm_provider == "zai"

    def test_get_error_class_rate_limit_error(self):
        """Test get_error_class for rate limit error"""
        config = ZaiChatConfig()
        
        result = config.get_error_class("Rate limit exceeded", 429, {})
        
        assert isinstance(result, litellm.RateLimitError)
        assert result.llm_provider == "zai"

    def test_get_error_class_generic_api_error(self):
        """Test get_error_class for generic API error"""
        config = ZaiChatConfig()
        
        result = config.get_error_class("Server error", 500, {})
        
        assert isinstance(result, litellm.APIError)
        assert result.status_code == 500
        assert result.llm_provider == "zai"


class TestZaiIntegration:
    """Integration tests for Zai provider"""

    @patch('litellm.main.base_llm_http_handler.completion')
    def test_zai_completion_call(self, mock_completion):
        """Test completion call using zai provider"""
        # Mock the response
        mock_response = ModelResponse(
            id="test-id",
            model="glm-4.6",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="Test response"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=15, total_tokens=25)
        )
        mock_completion.return_value = mock_response

        response = litellm.completion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key"
        )

        # Verify the mock was called with correct parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args.kwargs["custom_llm_provider"] == "zai"
        assert call_args.kwargs["model"] == "zai/glm-4.6"
        assert call_args.kwargs["api_key"] == "test-key"
        
        # Verify response structure
        assert response.id == "test-id"
        assert response.model == "glm-4.6"
        assert response.choices[0].message.content == "Test response"

    def test_zai_model_price_registration(self):
        """Test that zai models can be registered with custom pricing"""
        litellm.register_model({
            "zai/glm-4.6": {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "litellm_provider": "zai"
            }
        })
        
        # Verify the model is in the cost map
        assert "zai/glm-4.6" in litellm.model_cost
        model_info = litellm.model_cost["zai/glm-4.6"]
        assert model_info["input_cost_per_token"] == 0.001
        assert model_info["output_cost_per_token"] == 0.002
        assert model_info["litellm_provider"] == "zai"


def test_zai_provider_in_constants():
    """Test that zai is properly registered in constants"""
    from litellm.constants import LITELLM_CHAT_PROVIDERS
    
    assert "zai" in LITELLM_CHAT_PROVIDERS


def test_zai_config_import():
    """Test that ZaiChatConfig can be imported from main package"""
    from litellm import ZaiChatConfig
    
    config = ZaiChatConfig()
    assert hasattr(config, 'validate_environment')
    assert hasattr(config, 'get_complete_url')
    assert hasattr(config, 'transform_request')
    assert hasattr(config, 'transform_response')


if __name__ == "__main__":
    # Run basic validation tests
    test_config = TestZaiChatConfig()
    
    print("Testing validate_environment with API key...")
    test_config.test_validate_environment_with_api_key()
    print("✅ Passed")
    
    print("Testing get_complete_url...")
    test_config.test_get_complete_url_default()
    print("✅ Passed")
    
    print("Testing transform_request...")
    test_config.test_transform_request()
    print("✅ Passed")
    
    print("Testing transform_response...")
    test_config.test_transform_response()
    print("✅ Passed")
    
    print("Testing error class mapping...")
    test_config.test_get_error_class_authentication_error()
    print("✅ Passed")
    
    print("Testing provider constants...")
    test_zai_provider_in_constants()
    print("✅ Passed")
    
    print("Testing imports...")
    test_zai_config_import()
    print("✅ Passed")
    
    print("\n🎉 All Zai provider tests passed!")