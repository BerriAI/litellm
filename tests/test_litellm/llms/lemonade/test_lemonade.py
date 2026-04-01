import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.lemonade.chat.transformation import LemonadeChatConfig
from litellm.types.utils import ModelResponse
import httpx


def test_lemonade_config_initialization():
    """Test that LemonadeChatConfig can be initialized with various parameters"""
    config = LemonadeChatConfig(
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        top_k=50,
        repeat_penalty=1.1
    )
    
    assert config.custom_llm_provider == "lemonade"
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.top_p == 0.9
    assert config.top_k == 50
    assert config.repeat_penalty == 1.1


def test_get_openai_compatible_provider_info():
    """Test the provider info method returns correct API base and key"""
    config = LemonadeChatConfig()
    
    api_base, key = config._get_openai_compatible_provider_info(
        api_base=None, 
        api_key=None
    )
    
    assert api_base == "http://localhost:8000/api/v1"
    assert key == "lemonade"


def test_get_openai_compatible_provider_info_with_custom_base():
    """Test the provider info method with custom API base"""
    config = LemonadeChatConfig()
    
    custom_api_base = "https://custom.lemonade.ai/v1"
    api_base, key = config._get_openai_compatible_provider_info(
        api_base=custom_api_base, 
        api_key=None
    )
    
    assert api_base == custom_api_base
    assert key == "lemonade"


def test_transform_response():
    """Test the response transformation adds lemonade prefix to model name"""
    config = LemonadeChatConfig()
    
    # Mock raw response
    raw_response = MagicMock()
    raw_response.status_code = 200
    raw_response.headers = {}
    
    # Create a model response
    model_response = ModelResponse()
    
    # Mock the parent class transform_response method
    with patch.object(config.__class__.__bases__[0], 'transform_response') as mock_parent:
        mock_parent.return_value = model_response
        
        result = config.transform_response(
            model="test-model",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key="test-key",
            json_mode=False,
        )
        
        # Check that the model name is prefixed with "lemonade/"
        assert hasattr(result, 'model')
        assert result.model == "lemonade/test-model"


def test_config_get_config():
    """Test that get_config method returns the configuration"""
    config_dict = LemonadeChatConfig.get_config()
    assert isinstance(config_dict, dict)


def test_response_format_support():
    """Test that response_format parameter is supported"""
    response_format = {
        "type": "json_object"
    }
    
    config = LemonadeChatConfig(response_format=response_format)
    assert config.response_format == response_format


def test_tools_support():
    """Test that tools parameter is supported"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information"
            }
        }
    ]
    
    config = LemonadeChatConfig(tools=tools)
    assert config.tools == tools


def test_functions_support():
    """Test that functions parameter is supported"""
    functions = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    ]
    
    config = LemonadeChatConfig(functions=functions)
    assert config.functions == functions


def test_stop_parameter_support():
    """Test that stop parameter supports both string and list"""
    # Test with string
    config1 = LemonadeChatConfig(stop="STOP")
    assert config1.stop == "STOP"
    
    # Test with list
    config2 = LemonadeChatConfig(stop=["STOP", "END"])
    assert config2.stop == ["STOP", "END"]


def test_logit_bias_support():
    """Test that logit_bias parameter is supported"""
    logit_bias = {"50256": -100}
    
    config = LemonadeChatConfig(logit_bias=logit_bias)
    assert config.logit_bias == logit_bias


def test_presence_penalty_support():
    """Test that presence_penalty parameter is supported"""
    config = LemonadeChatConfig(presence_penalty=0.5)
    assert config.presence_penalty == 0.5


def test_n_parameter_support():
    """Test that n parameter (number of completions) is supported"""
    config = LemonadeChatConfig(n=3)
    assert config.n == 3


def test_max_completion_tokens_support():
    """Test that max_completion_tokens parameter is supported"""
    config = LemonadeChatConfig(max_completion_tokens=150)
    assert config.max_completion_tokens == 150