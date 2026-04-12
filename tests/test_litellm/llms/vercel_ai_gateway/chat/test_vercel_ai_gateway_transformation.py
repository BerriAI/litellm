import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.vercel_ai_gateway.chat.transformation import (
    VercelAIGatewayConfig,
)
from litellm.llms.vercel_ai_gateway.common_utils import VercelAIGatewayException


def test_vercel_ai_gateway_extra_body_transformation():
    """Test that providerOptions is correctly moved to extra_body"""
    transformed_request = VercelAIGatewayConfig().transform_request(
        model="vercel_ai_gateway/openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        optional_params={
            "extra_body": {
                "providerOptions": {
                    "gateway": {"order": ["azure", "openai"]}
                }
            }
        },
        litellm_params={},
        headers={},
    )

    assert transformed_request["extra_body"]["providerOptions"]["gateway"]["order"] == ["azure", "openai"]
    assert transformed_request["messages"] == [
        {"role": "user", "content": "Hello, world!"}
    ]


def test_vercel_ai_gateway_provider_options_mapping():
    """Test that providerOptions from non_default_params is moved to extra_body"""
    config = VercelAIGatewayConfig()
    
    non_default_params = {
        "providerOptions": {
            "gateway": {"order": ["azure", "openai"]}
        }
    }
    optional_params = {}
    model = "vercel_ai_gateway/openai/gpt-4o"
    
    result = config.map_openai_params(
        non_default_params, optional_params, model, drop_params=False
    )
    
    assert result["extra_body"]["providerOptions"]["gateway"]["order"] == ["azure", "openai"]
    assert "providerOptions" not in result


def test_vercel_ai_gateway_get_supported_openai_params():
    """Test that extra_body is included in supported params"""
    config = VercelAIGatewayConfig()
    supported_params = config.get_supported_openai_params("vercel_ai_gateway/openai/gpt-4o")
    
    assert "extra_body" in supported_params
    assert "temperature" in supported_params
    assert "max_tokens" in supported_params
    assert "stream" in supported_params


def test_vercel_ai_gateway_get_openai_compatible_provider_info():
    """Test provider info retrieval with environment variables"""
    config = VercelAIGatewayConfig()
    
    with patch.dict(
        "os.environ",
        {
            "VERCEL_AI_GATEWAY_API_BASE": "https://env.vercel.sh/v1",
            "VERCEL_AI_GATEWAY_API_KEY": "env_api_key",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://env.vercel.sh/v1"
        assert api_key == "env_api_key"


def test_vercel_ai_gateway_error_class():
    """Test error class creation"""
    config = VercelAIGatewayConfig()
    
    error_message = "Test error"
    status_code = 400
    headers = {"Content-Type": "application/json"}
    
    error_class = config.get_error_class(error_message, status_code, headers)
    
    assert isinstance(error_class, VercelAIGatewayException)
    assert error_class.message == error_message
    assert error_class.status_code == status_code
    assert error_class.headers == headers


def test_vercel_ai_gateway_model_name_preserved_in_response():
    """Test that multi-segment provider prefixes like 'vercel_ai_gateway/anthropic' are
    preserved when reconstructing the model name from the API response.

    Regression test for https://github.com/BerriAI/litellm/issues/20412
    """
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )
    from litellm.types.utils import ModelResponse

    # Simulate: preset model is "vercel_ai_gateway/anthropic/claude-opus-4.5"
    # and the Vercel API response returns model="claude-opus-4.5"
    model_response = ModelResponse()
    model_response.model = "vercel_ai_gateway/anthropic/claude-opus-4.5"

    response_object = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "claude-opus-4.5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    result = convert_to_model_response_object(
        response_object=response_object,
        model_response_object=model_response,
    )

    # The full provider prefix must be preserved
    assert result.model == "vercel_ai_gateway/anthropic/claude-opus-4.5", (
        f"Expected 'vercel_ai_gateway/anthropic/claude-opus-4.5', got '{result.model}'"
    )


def test_vercel_ai_gateway_single_segment_provider_still_works():
    """Verify that single-segment providers like 'openai/gpt-4o' still work
    after the multi-segment fix."""
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )
    from litellm.types.utils import ModelResponse

    model_response = ModelResponse()
    model_response.model = "openai/gpt-4o"

    response_object = {
        "id": "chatcmpl-xyz789",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        },
    }

    result = convert_to_model_response_object(
        response_object=response_object,
        model_response_object=model_response,
    )

    assert result.model == "openai/gpt-4o-2024-08-06", (
        f"Expected 'openai/gpt-4o-2024-08-06', got '{result.model}'"
    )


def test_vercel_ai_gateway_exception_inheritance():
    """Test that VercelAIGatewayException inherits from BaseLLMException"""
    from litellm.llms.base_llm.chat.transformation import BaseLLMException
    
    exception = VercelAIGatewayException(
        message="test", 
        status_code=500, 
        headers={}
    )
    
    assert isinstance(exception, BaseLLMException)
