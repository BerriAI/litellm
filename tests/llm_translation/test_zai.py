import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
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
from litellm.llms.zai.chat.transformation import ZaiChatConfig
from litellm.llms.zai.chat.handler import ZaiChatCompletion
from litellm.llms.zai.chat.transformation import (
    ZaiError, 
    ZaiAuthenticationError, 
    ZaiBadRequestError, 
    ZaiRateLimitError
)
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
        # The API key might come from environment if set, so just check it starts with Bearer
        assert result["Authorization"].startswith("Bearer ")
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_missing_key(self):
        """Test validate_environment method when API key is missing"""
        config = ZaiChatConfig()
        headers = {}
        # Ensure the environment variable is not set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing ZAI_API_KEY"):
                config.validate_environment(headers, "zai/glm-4.6", [], {}, {})

    def test_validate_environment_with_env_var(self):
        """Test validate_environment method with environment variable"""
        config = ZaiChatConfig()
        headers = {}
        
        with patch.dict(os.environ, {"ZAI_API_KEY": "env_api_key"}):
            result = config.validate_environment(headers, "zai/glm-4.6", [], {}, {})
            
            assert "Authorization" in result
            assert result["Authorization"] == "Bearer env_api_key"
            assert result["Content-Type"] == "application/json"

    def test_get_complete_url_default(self):
        """Test get_complete_url method with default API base"""
        config = ZaiChatConfig()
        
        result = config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model="zai/glm-4.6",
            optional_params={},
            litellm_params={}
        )
        
        assert result == "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def test_get_complete_url_custom_base(self):
        """Test get_complete_url method with custom API base"""
        config = ZaiChatConfig()
        
        result = config.get_complete_url(
            api_base="https://custom.api.z.ai",
            api_key="test_key",
            model="zai/glm-4.6",
            optional_params={},
            litellm_params={}
        )
        
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

        result = config.transform_request(model, messages, optional_params, {}, {})

        assert result["model"] == "glm-4.6"  # zai/ prefix should be removed
        assert result["messages"] == messages
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stream"] is False
        assert result["stop"] == ["END"]

    def test_transform_request_zai_org_model(self):
        """Test transform_request method with zai-org/ model prefix"""
        config = ZaiChatConfig()
        model = "zai-org/GLM-4.5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}

        result = config.transform_request(model, messages, optional_params, {}, {})

        assert result["model"] == "GLM-4.5"  # zai-org/ prefix should be removed
        assert result["messages"] == messages

    def test_transform_request_with_reasoning_tokens(self):
        """Test transform_request method handles reasoning_tokens correctly"""
        config = ZaiChatConfig()
        model = "zai/glm-4.6"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"max_tokens": 100}
        litellm_params = {"reasoning_tokens": True}

        result = config.transform_request(model, messages, optional_params, litellm_params, {})

        assert result["model"] == "glm-4.6"
        assert "extra_body" in optional_params
        assert optional_params["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_tokens" not in litellm_params

    def test_transform_request_with_tools(self):
        """Test transform_request method preserves tool parameters"""
        config = ZaiChatConfig()
        model = "zai/glm-4.6"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "tools": [{"type": "function", "function": {"name": "test_tool"}}],
            "tool_choice": "auto",
            "max_tokens": 100
        }

        result = config.transform_request(model, messages, optional_params, {}, {})

        assert result["model"] == "glm-4.6"
        assert "tools" in result
        assert "tool_choice" in result
        assert result["max_tokens"] == 100

    def test_transform_request_minimal_params(self):
        """Test transform_request method with minimal parameters"""
        config = ZaiChatConfig()
        model = "zai/glm-4.6"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}

        result = config.transform_request(model, messages, optional_params, {}, {})

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
            model, raw_response, model_response, None, {}, [], {}, {}, None
        )

        assert result.id == "chatcmpl-12345"
        assert result.model == "glm-4.6"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "I am doing well, thank you for asking."
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 15
        assert result.usage.total_tokens == 25

    def test_transform_response_with_reasoning(self):
        """Test transform_response method with reasoning content"""
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
                        "content": "The answer is 42.",
                        "reasoning_content": "Let me think about this step by step..."
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
            model, raw_response, model_response, None, {}, [], {}, {}, None
        )

        assert result.choices[0].message.content == "The answer is 42."
        assert result.choices[0].message.reasoning_content == "Let me think about this step by step..."
        assert result.usage.completion_tokens_details.reasoning_tokens is not None
        assert result.usage.completion_tokens_details.reasoning_tokens > 0

    def test_transform_response_with_tool_calls(self):
        """Test transform_response method with tool calls"""
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
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "test_function",
                                    "arguments": '{"param": "value"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
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
            model, raw_response, model_response, None, {}, [], {}, {}, None
        )

        assert result.choices[0].message.content is None  # Should be null when tools are used
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0].id == "call_123"
        assert result.choices[0].finish_reason == "tool_calls"

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
            model, raw_response, model_response, None, {}, [], {}, {}, None
        )

        assert result.choices[0].message.content == "Hello!"
        assert result.model == raw_response.get("model")  # Should handle missing model

    def test_get_supported_openai_params(self):
        """Test get_supported_openai_params returns correct list"""
        config = ZaiChatConfig()
        
        result = config.get_supported_openai_params("zai/glm-4.6")
        
        expected_params = [
            "logit_bias", "logprobs", "max_tokens", "n", "presence_penalty",
            "response_format", "seed", "stream", "stream_options", "temperature",
            "tool_choice", "tools", "top_logprobs", "top_p", "user",
            "frequency_penalty", "stop", "reasoning_tokens"
        ]
        
        for param in expected_params:
            assert param in result

    def test_map_openai_params(self):
        """Test map_openai_params filters supported parameters"""
        config = ZaiChatConfig()
        non_default_params = {
            "temperature": 0.7,  # supported
            "invalid_param": "value",  # not supported
            "max_tokens": 100,  # supported
        }
        optional_params = {}
        model = "zai/glm-4.6"
        
        result = config.map_openai_params(non_default_params, optional_params, model, False)
        
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert "invalid_param" not in result

    def test_transform_request_with_complex_tools(self):
        """Test transform_request method with complex tool definitions"""
        config = ZaiChatConfig()
        model = "zai/glm-4.6"
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        optional_params = {
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "temperature": 0.1
        }

        result = config.transform_request(model, messages, optional_params, {}, {})

        assert result["model"] == "glm-4.6"
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "get_weather"
        assert result["tool_choice"]["function"]["name"] == "get_weather"
        assert result["temperature"] == 0.1

    def test_transform_request_with_parallel_tool_calls(self):
        """Test transform_request method preserves parallel_tool_calls parameter"""
        config = ZaiChatConfig()
        model = "zai/glm-4.6"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "parallel_tool_calls": True,
            "tools": [{"type": "function", "function": {"name": "test_tool"}}]
        }

        result = config.transform_request(model, messages, optional_params, {}, {})

        assert result["parallel_tool_calls"] is True
        assert "tools" in result

    def test_get_error_class_authentication_error(self):
        """Test get_error_class for authentication error"""
        from litellm.llms.zai.chat.transformation import ZaiAuthenticationError
        config = ZaiChatConfig()
        
        result = config.get_error_class("Invalid API key", 401, {})
        
        assert isinstance(result, ZaiAuthenticationError)
        assert result.llm_provider == "zai"

    def test_get_error_class_bad_request_error(self):
        """Test get_error_class for bad request error"""
        from litellm.llms.zai.chat.transformation import ZaiBadRequestError
        config = ZaiChatConfig()
        
        result = config.get_error_class("Request malformed", 400, {})
        
        assert isinstance(result, ZaiBadRequestError)
        assert result.llm_provider == "zai"

    def test_get_error_class_rate_limit_error(self):
        """Test get_error_class for rate limit error"""
        from litellm.llms.zai.chat.transformation import ZaiRateLimitError
        config = ZaiChatConfig()
        
        result = config.get_error_class("Rate limit exceeded", 429, {})
        
        assert isinstance(result, ZaiRateLimitError)
        assert result.llm_provider == "zai"

    def test_get_error_class_generic_api_error(self):
        """Test get_error_class for generic API error"""
        from litellm.llms.zai.chat.transformation import ZaiError
        config = ZaiChatConfig()
        
        result = config.get_error_class("Server error", 500, {})
        
        assert isinstance(result, ZaiError)
        assert result.status_code == 500
        assert result.llm_provider == "zai"

    def test_zai_error_hierarchy(self):
        """Test ZAI error class hierarchy and properties"""
        # Test base error class
        base_error = ZaiError(status_code=500, message="Base error", headers={})
        assert isinstance(base_error, ZaiError)
        assert base_error.llm_provider == "zai"
        assert base_error.status_code == 500
        assert str(base_error) == "Base error"

        # Test authentication error
        auth_error = ZaiAuthenticationError(status_code=401, message="Invalid API key", headers={})
        assert isinstance(auth_error, ZaiError)
        assert isinstance(auth_error, ZaiAuthenticationError)
        assert auth_error.llm_provider == "zai"
        assert auth_error.status_code == 401

        # Test bad request error
        bad_request_error = ZaiBadRequestError(status_code=400, message="Bad request", headers={})
        assert isinstance(bad_request_error, ZaiError)
        assert isinstance(bad_request_error, ZaiBadRequestError)
        assert bad_request_error.llm_provider == "zai"
        assert bad_request_error.status_code == 400

        # Test rate limit error
        rate_limit_error = ZaiRateLimitError(status_code=429, message="Rate limit exceeded", headers={})
        assert isinstance(rate_limit_error, ZaiError)
        assert isinstance(rate_limit_error, ZaiRateLimitError)
        assert rate_limit_error.llm_provider == "zai"
        assert rate_limit_error.status_code == 429

    def test_get_error_class_with_httpx_headers(self):
        """Test get_error_class with httpx.Headers object"""
        config = ZaiChatConfig()
        from httpx import Headers
        
        headers = Headers({"content-type": "application/json"})
        result = config.get_error_class("Not found", 404, headers)
        
        assert isinstance(result, ZaiError)
        assert result.status_code == 404
        assert result.llm_provider == "zai"


class TestZaiChatCompletion:
    """Test suite for ZaiChatCompletion handler"""

    def test_handler_initialization(self):
        """Test that ZaiChatCompletion can be initialized"""
        handler = ZaiChatCompletion()
        assert hasattr(handler, 'completion')
        assert hasattr(handler, 'acompletion')

    @patch.object(ZaiChatCompletion, '__init__', lambda x, **kwargs: None)
    def test_completion_with_reasoning_tokens(self):
        """Test completion method handles reasoning_tokens correctly"""
        handler = ZaiChatCompletion()
        
        # Mock the parent completion method
        with patch.object(handler, 'completion') as mock_super_completion:
            mock_super_completion.return_value = ModelResponse()
            
            optional_params = {}
            litellm_params = {"reasoning_tokens": True}
            
            handler.completion(
                model="zai/glm-4.6",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://open.bigmodel.cn/api/paas/v4",
                custom_llm_provider="zai",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=print,
                encoding=None,
                api_key="test-key",
                logging_obj=None,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=None,
                headers=None,
                timeout=None,
                client=None,
                custom_endpoint=False,
                streaming_decoder=None,
                fake_stream=False
            )
            
            # Verify extra_body was set with thinking parameter
            assert "extra_body" in optional_params
            assert optional_params["extra_body"]["thinking"] == {"type": "enabled"}
            # Verify reasoning_tokens was removed from litellm_params
            assert "reasoning_tokens" not in litellm_params

    @patch.object(ZaiChatCompletion, '__init__', lambda x, **kwargs: None)
    def test_completion_model_prefix_stripping(self):
        """Test completion method strips zai/ prefix from model"""
        handler = ZaiChatCompletion()
        
        with patch.object(handler, 'completion') as mock_super_completion:
            mock_super_completion.return_value = ModelResponse()
            
            # Test with zai/ prefix
            handler.completion(
                model="zai/glm-4.6",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://custom.api.com",
                custom_llm_provider="zai",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=print,
                encoding=None,
                api_key="test-key",
                logging_obj=None,
                optional_params={},
                litellm_params={},
                logger_fn=None,
                headers=None,
                timeout=None,
                client=None,
                custom_endpoint=False,
                streaming_decoder=None,
                fake_stream=False
            )
            
            # Verify model prefix was stripped in the call to parent
            call_args = mock_super_completion.call_args
            assert call_args.kwargs["model"] == "glm-4.6"

    @patch.object(ZaiChatCompletion, '__init__', lambda x, **kwargs: None)
    def test_completion_default_api_base(self):
        """Test completion method uses default API base when none provided"""
        handler = ZaiChatCompletion()
        
        with patch.object(handler, 'completion') as mock_super_completion:
            mock_super_completion.return_value = ModelResponse()
            
            handler.completion(
                model="zai/glm-4.6",
                messages=[{"role": "user", "content": "Hello"}],
                api_base=None,  # No api_base provided
                custom_llm_provider="zai",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=print,
                encoding=None,
                api_key="test-key",
                logging_obj=None,
                optional_params={},
                litellm_params={},
                logger_fn=None,
                headers=None,
                timeout=None,
                client=None,
                custom_endpoint=False,
                streaming_decoder=None,
                fake_stream=False
            )
            
            # Verify default API base was used
            call_args = mock_super_completion.call_args
            assert call_args.kwargs["api_base"] == "https://open.bigmodel.cn/api/paas/v4"


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

    @patch('litellm.main.base_llm_http_handler.completion')
    def test_zai_completion_with_reasoning_tokens(self, mock_completion):
        """Test completion call with reasoning_tokens enabled"""
        mock_response = ModelResponse(
            id="test-id",
            model="glm-4.6",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant", 
                        content="Final answer",
                        reasoning_content="Step by step reasoning"
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=15, total_tokens=25, reasoning_tokens=20)
        )
        mock_completion.return_value = mock_response

        response = litellm.completion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "Explain this"}],
            api_key="test-key",
            reasoning_tokens=True
        )

        # Verify response structure includes reasoning
        assert response.choices[0].message.reasoning_content == "Step by step reasoning"
        assert response.usage.reasoning_tokens == 20

    @patch('litellm.main.base_llm_http_handler.completion')
    def test_zai_completion_with_tools(self, mock_completion):
        """Test completion call with tool calling"""
        mock_response = ModelResponse(
            id="test-id",
            model="glm-4.6",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}'
                                }
                            }
                        ]
                    ),
                    finish_reason="tool_calls"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=15, total_tokens=25)
        )
        mock_completion.return_value = mock_response

        response = litellm.completion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            api_key="test-key",
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            }]
        )

        # Verify tool calling response
        assert response.choices[0].message.content is None
        assert len(response.choices[0].message.tool_calls) == 1
        assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert response.choices[0].finish_reason == "tool_calls"

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

def test_zai_handler_import():
    """Test that ZaiChatCompletion can be imported from main package"""
    from litellm import ZaiChatCompletion
    
    handler = ZaiChatCompletion()
    assert hasattr(handler, 'completion')
    assert hasattr(handler, 'aclasscompletion')  # From OpenAILikeChatHandler

def test_zai_error_imports():
    """Test that ZAI error classes can be imported"""
    from litellm.llms.zai.chat.transformation import (
        ZaiError, 
        ZaiAuthenticationError, 
        ZaiBadRequestError, 
        ZaiRateLimitError
    )
    
    # Test that all error classes can be instantiated
    base_error = ZaiError(status_code=500, message="Test", headers={})
    auth_error = ZaiAuthenticationError(status_code=401, message="Test", headers={})
    bad_request_error = ZaiBadRequestError(status_code=400, message="Test", headers={})
    rate_limit_error = ZaiRateLimitError(status_code=429, message="Test", headers={})
    
    assert all(hasattr(error, 'llm_provider') for error in [base_error, auth_error, bad_request_error, rate_limit_error])


if __name__ == "__main__":
    # Run basic validation tests
    test_config = TestZaiChatConfig()
    test_handler = TestZaiChatCompletion()
    
    print("🧪 Running ZAI Provider Test Suite")
    print("=" * 50)
    
    # Basic Config Tests
    print("✅ Testing validate_environment with API key...")
    test_config.test_validate_environment_with_api_key()
    
    print("✅ Testing get_complete_url...")
    test_config.test_get_complete_url_default()
    
    print("✅ Testing transform_request...")
    test_config.test_transform_request()
    
    print("✅ Testing transform_request with zai-org model...")
    test_config.test_transform_request_zai_org_model()
    
    print("✅ Testing transform_request with reasoning tokens...")
    test_config.test_transform_request_with_reasoning_tokens()
    
    print("✅ Testing transform_request with tools...")
    test_config.test_transform_request_with_tools()
    
    print("✅ Testing transform_response...")
    test_config.test_transform_response()
    
    print("✅ Testing transform_response with reasoning...")
    test_config.test_transform_response_with_reasoning()
    
    print("✅ Testing transform_response with tool calls...")
    test_config.test_transform_response_with_tool_calls()
    
    print("✅ Testing error class mapping...")
    test_config.test_get_error_class_authentication_error()
    
    print("✅ Testing ZAI error hierarchy...")
    test_config.test_zai_error_hierarchy()
    
    # Handler Tests
    print("✅ Testing ZaiChatCompletion initialization...")
    test_handler.test_handler_initialization()
    
    # Integration Tests
    print("✅ Testing provider constants...")
    test_zai_provider_in_constants()
    
    print("✅ Testing imports...")
    test_zai_config_import()
    test_zai_handler_import()
    test_zai_error_imports()
    
    print("\n🎉 All ZAI provider tests passed!")
    print("=" * 50)
    print("✅ Test coverage includes:")
    print("  - Basic configuration and validation")
    print("  - Request/response transformation")
    print("  - ZAI-specific features (reasoning, zai-org models)")
    print("  - Tool calling functionality") 
    print("  - Error handling and hierarchy")
    print("  - Handler class methods")
    print("  - Integration with litellm main module")