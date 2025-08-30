import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_alibaba_provider_routing():
    """Test that alibaba provider is properly routed"""
    model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
        model="alibaba/qwen3-coder-plus", 
        custom_llm_provider=None,
        api_base=None,
        api_key=None
    )
    
    assert model == "qwen3-coder-plus"
    assert custom_llm_provider == "alibaba"
    assert api_base == "https://portal.qwen.ai/v1"


def test_alibaba_in_provider_lists():
    """Test that alibaba is registered in all necessary provider lists"""
    assert "alibaba" in litellm.openai_compatible_providers
    assert "alibaba" in litellm.provider_list
    assert "https://portal.qwen.ai/v1" in litellm.openai_compatible_endpoints


@pytest.mark.asyncio
async def test_alibaba_completion_call():
    """Test completion call with Alibaba provider (requires ALIBABA_API_KEY)"""
    # Skip if no API key is available
    if not os.getenv("ALIBABA_API_KEY"):
        pytest.skip("ALIBABA_API_KEY not set")
    
    try:
        response = await litellm.acompletion(
            model="alibaba/qwen3-coder-plus",
            messages=[{"role": "user", "content": "Hello, what model are you?"}],
            max_tokens=50,
        )
        
        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert "qwen" in response.choices[0].message.content.lower()
        
    except Exception as e:
        if "invalid access token" in str(e):
            pytest.skip(f"Skipping due to auth issue: {e}")
        else:
            raise e


def test_alibaba_model_configuration():
    """Test that alibaba models have correct configuration"""
    model_info = litellm.get_model_info("alibaba/qwen3-coder-plus")
    
    assert model_info is not None
    assert model_info["litellm_provider"] == "alibaba"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_tool_choice"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["max_tokens"] == 131072
    assert model_info["max_input_tokens"] == 131072
    assert model_info["max_output_tokens"] == 8192


def test_alibaba_chat_config():
    """Test that AlibabaChatConfig is properly configured"""
    from litellm.llms.alibaba.chat.transformation import AlibabaChatConfig
    
    # Test config class instantiation
    config = AlibabaChatConfig()
    assert config is not None
    
    # Test get_openai_compatible_provider_info method
    api_base, dynamic_api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=None
    )
    assert api_base == "https://portal.qwen.ai/v1"
    assert dynamic_api_key is None  # No key provided
    
    # Test API base configuration
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == "https://portal.qwen.ai/v1"


def test_alibaba_cost_calculation():
    """Test that alibaba cost calculation works"""
    from litellm.llms.alibaba.cost_calculator import cost_per_token
    from litellm.types.utils import Usage
    
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    
    input_cost, output_cost = cost_per_token("alibaba/qwen3-coder-plus", usage)
    
    # Based on the model configuration: input_cost_per_token=1e-06, output_cost_per_token=5e-06
    expected_input_cost = 100 * 1e-06  # 100 tokens * $0.000001
    expected_output_cost = 50 * 5e-06   # 50 tokens * $0.000005
    
    assert abs(input_cost - expected_input_cost) < 1e-10
    assert abs(output_cost - expected_output_cost) < 1e-10


@pytest.mark.parametrize("model_name", [
    "alibaba/qwen3-coder-plus",
])
def test_alibaba_models_in_model_list(model_name):
    """Test that alibaba models are properly registered"""
    assert model_name in litellm.model_list


def test_alibaba_endpoint_configuration():
    """Test that alibaba endpoint is properly configured"""
    from litellm.constants import openai_compatible_endpoints, openai_compatible_providers
    
    assert "alibaba" in openai_compatible_providers
    assert "https://portal.qwen.ai/v1" in openai_compatible_endpoints


if __name__ == "__main__":
    # Run basic tests that don't require API key
    test_alibaba_provider_routing()
    test_alibaba_in_provider_lists()
    test_alibaba_model_configuration()
    test_alibaba_chat_config()
    test_alibaba_cost_calculation()
    test_alibaba_models_in_model_list("alibaba/qwen3-coder-plus")
    test_alibaba_endpoint_configuration()
    
    print("✅ All basic alibaba provider tests passed!")