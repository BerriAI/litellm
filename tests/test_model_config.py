import pytest
import json

def test_gpt_5_search_api_config():
    """Test that gpt-5-search-api model is properly configured"""
    with open('model_prices_and_context_window.json', 'r') as f:
        config = json.load(f)
    
    model = config.get('gpt-5-search-api')
    
    # Check model exists
    assert model is not None, "gpt-5-search-api not found in config"
    
    # Check required fields
    assert model['litellm_provider'] == 'openai'
    assert model['input_cost_per_token'] == 1.25e-06
    assert model['output_cost_per_token'] == 1e-05
    assert model['max_input_tokens'] == 400000
    assert model['max_output_tokens'] == 128000
    assert model['mode'] == 'chat'
    assert model['supports_function_calling'] == True
    
    print("✅ gpt-5-search-api config is valid!")
