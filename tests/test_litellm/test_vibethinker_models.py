"""
Test VibeThinker models configuration.
"""
import json
import os
import pytest


def test_vibethinker_models_in_model_prices():
    """Verify VibeThinker models are correctly configured in model_prices_and_context_window.json."""
    # Use absolute path relative to test file
    model_prices_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "model_prices_and_context_window.json"
    )
    with open(model_prices_path, "r") as f:
        model_prices = json.load(f)
    
    models = [
        "featherless_ai/WeiboAI/VibeThinker-1.5B",
        "featherless_ai/WeiboAI/VibeThinker-3B",
    ]
    
    for model in models:
        assert model in model_prices, f"{model} not found in model_prices_and_context_window.json"
        
        config = model_prices[model]
        
        # Verify required fields
        assert config["litellm_provider"] == "featherless_ai", f"{model}: wrong provider"
        assert config["mode"] == "chat", f"{model}: wrong mode"
        assert "max_tokens" in config, f"{model}: missing max_tokens"
        assert "max_input_tokens" in config, f"{model}: missing max_input_tokens"
        assert "max_output_tokens" in config, f"{model}: missing max_output_tokens"
        
        # Verify context window (Qwen base models have 32K input, 8K output)
        assert config["max_input_tokens"] == 32768, f"{model}: incorrect max_input_tokens"
        assert config["max_tokens"] == 8192, f"{model}: incorrect max_tokens (should match max_output_tokens)"


def test_vibethinker_provider_consistency():
    """Verify VibeThinker models use the same provider pattern as other Featherless AI models."""
    # Use absolute path relative to test file
    model_prices_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "model_prices_and_context_window.json"
    )
    with open(model_prices_path, "r") as f:
        model_prices = json.load(f)
    
    # Get all featherless_ai models
    featherless_models = {k: v for k, v in model_prices.items() if k.startswith("featherless_ai/")}
    
    # VibeThinker models should exist
    assert "featherless_ai/WeiboAI/VibeThinker-1.5B" in featherless_models
    assert "featherless_ai/WeiboAI/VibeThinker-3B" in featherless_models
    
    # All should have the same provider
    for model, config in featherless_models.items():
        assert config["litellm_provider"] == "featherless_ai", f"{model}: inconsistent provider"
