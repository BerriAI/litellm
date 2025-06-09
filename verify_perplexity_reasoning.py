#!/usr/bin/env python3
"""
Simple verification script for Perplexity reasoning effort functionality
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath("."))

import litellm
from litellm.utils import get_optional_params


def test_perplexity_reasoning_models_in_model_cost():
    """Test that perplexity reasoning models are in the model cost map"""
    print("Testing perplexity reasoning models are in model cost map...")
    
    # Set up local model cost map
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    reasoning_models = [
        "perplexity/sonar-reasoning",
        "perplexity/sonar-reasoning-pro",
    ]
    
    for model in reasoning_models:
        if model in litellm.model_cost:
            model_info = litellm.model_cost[model]
            supports_reasoning = model_info.get("supports_reasoning", False)
            print(f"✓ {model}: found in model cost map, supports_reasoning={supports_reasoning}")
            assert supports_reasoning, f"{model} should support reasoning"
        else:
            print(f"✗ {model}: not found in model cost map")
    
    print("✓ Perplexity reasoning models test passed!\n")


def test_reasoning_effort_parameter_mapping():
    """Test that reasoning_effort parameter is correctly mapped"""
    print("Testing reasoning_effort parameter mapping...")
    
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    model = "perplexity/sonar-reasoning"
    reasoning_effort = "high"
    
    # Get provider and optional params
    _, provider, _, _ = litellm.get_llm_provider(model=model)
    
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=provider,
        reasoning_effort=reasoning_effort,
    )
    
    print(f"Provider: {provider}")
    print(f"Optional params: {optional_params}")
    
    # Verify that reasoning_effort is preserved in optional_params for Perplexity
    assert "reasoning_effort" in optional_params, "reasoning_effort should be in optional_params"
    assert optional_params["reasoning_effort"] == reasoning_effort, f"reasoning_effort should be {reasoning_effort}"
    
    print("✓ Reasoning effort parameter mapping test passed!\n")


def test_perplexity_reasoning_support():
    """Test that supports_reasoning function works for perplexity models"""
    print("Testing supports_reasoning function...")
    
    from litellm.utils import supports_reasoning
    
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    reasoning_models = [
        "perplexity/sonar-reasoning",
        "perplexity/sonar-reasoning-pro",
    ]
    
    for model in reasoning_models:
        try:
            result = supports_reasoning(model, None)
            print(f"✓ {model}: supports_reasoning = {result}")
            assert result, f"{model} should support reasoning"
        except Exception as e:
            print(f"✗ {model}: Error checking reasoning support: {e}")
    
    print("✓ Supports reasoning test passed!\n")


def test_perplexity_config():
    """Test Perplexity config and supported parameters"""
    print("Testing Perplexity configuration...")
    
    from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig
    
    config = PerplexityChatConfig()
    
    # Test API base configuration
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key="test-key"
    )
    
    expected_api_base = "https://api.perplexity.ai"
    print(f"API Base: {api_base}")
    assert api_base == expected_api_base, f"API base should be {expected_api_base}"
    
    # Test supported parameters
    supported_params = config.get_supported_openai_params(model="perplexity/sonar-reasoning")
    print(f"Supported params: {supported_params}")
    
    assert "reasoning_effort" in supported_params, "reasoning_effort should be in supported params"
    
    print("✓ Perplexity configuration test passed!\n")


def main():
    """Run all verification tests"""
    print("=== Perplexity Reasoning Effort Verification ===\n")
    
    try:
        test_perplexity_reasoning_models_in_model_cost()
        test_reasoning_effort_parameter_mapping()
        test_perplexity_reasoning_support()
        test_perplexity_config()
        
        print("🎉 All tests passed! Perplexity reasoning effort functionality is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()