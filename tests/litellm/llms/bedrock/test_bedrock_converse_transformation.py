import pytest
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.types.llms.bedrock import ConverseTokenUsageBlock

def test_transform_request_helper_config_blocks():
    config = AmazonConverseConfig()
    model = "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Test serviceTier placement (Bug #17336)
    optional_params = {
        "serviceTier": {"type": "flex"},
        "guardrailConfig": {"guardrailIdentifier": "test", "guardrailVersion": "1"}
    }
    
    data = config._transform_request_helper(
        model=model,
        system_content_blocks=[],
        optional_params=optional_params
    )
    
    # Assert top-level placement
    assert "serviceTier" in data
    assert data["serviceTier"] == {"type": "flex"}
    assert "guardrailConfig" in data
    
    # Assert NOT in inferenceConfig
    assert "serviceTier" not in data["inferenceConfig"]
    assert "guardrailConfig" not in data["inferenceConfig"]

def test_transform_usage_recalculation():
    config = AmazonConverseConfig()
    usage_block = ConverseTokenUsageBlock(
        inputTokens=100,
        outputTokens=50,
        totalTokens=150,
        cacheReadInputTokens=20
    )
    
    usage = config._transform_usage(usage_block)
    
    # Assert input_tokens includes cache tokens
    assert usage.prompt_tokens == 120
    # Assert total_tokens is recalculated (Bug 2)
    assert usage.total_tokens == 170

def test_map_openai_params_stop_filtering():
    config = AmazonConverseConfig()
    optional_params = {}
    
    # Test empty list (Bug 3)
    config.map_openai_params(
        non_default_params={"stop": []},
        optional_params=optional_params,
        model="bedrock/anthropic.claude-3",
        drop_params=False
    )
    assert "stopSequences" not in optional_params
    
    # Test list with empty strings
    config.map_openai_params(
        non_default_params={"stop": ["", "valid"]},
        optional_params=optional_params,
        model="bedrock/anthropic.claude-3",
        drop_params=False
    )
    assert optional_params["stopSequences"] == ["valid"]

        # Test valid string stop - this would fail if stop block is outside the for loop
    config.map_openai_params(
        non_default_params={"stop": "END"},
        optional_params=optional_params,
        model="bedrock/anthropic.claude-3",
        drop_params=False
    )
    assert optional_params["stopSequences"] == ["END"]

    # Test valid list of stop strings
    config.map_openai_params(
        non_default_params={"stop": ["STOP1", "STOP2"]},
        optional_params=optional_params,
        model="bedrock/anthropic.claude-3",
        drop_params=False
    )
    assert optional_params["stopSequences"] == ["STOP1", "STOP2"]


def test_map_openai_params_multi_param_with_stop():
    """Test that stop param works correctly when passed alongside other params.
    This catches the indentation bug where stop block was outside the for loop.
    """
    config = AmazonConverseConfig()
    optional_params = {}

    # Pass temperature, stop, and top_p all at once
    # If stop block is outside for loop, temperature and top_p won't be mapped
    config.map_openai_params(
        non_default_params={"temperature": 0.7, "stop": ["END", "STOP"], "top_p": 0.9},
        optional_params=optional_params,
        model="bedrock/anthropic.claude-3",
        drop_params=False,
    )
    # All three params must be present - if stop block breaks the loop,
    # temperature and/or top_p will be missing
    assert optional_params["temperature"] == 0.7
    assert optional_params["stopSequences"] == ["END", "STOP"]
    assert optional_params["topP"] == 0.9

    # Test with empty stop list alongside other params
    optional_params2 = {}
    config.map_openai_params(
        non_default_params={"temperature": 0.5, "stop": [], "top_p": 0.8},
        optional_params=optional_params2,
        model="bedrock/anthropic.claude-3",
        drop_params=False,
    )
    assert optional_params2["temperature"] == 0.5
    assert "stopSequences" not in optional_params2  # Empty stop list filtered out
    assert optional_params2["topP"] == 0.8

    # Test with string stop alongside other params
    optional_params3 = {}
    config.map_openai_params(
        non_default_params={"temperature": 0.3, "stop": "HALT", "top_p": 0.95},
        optional_params=optional_params3,
        model="bedrock/anthropic.claude-3",
        drop_params=False,
    )
    assert optional_params3["temperature"] == 0.3
    assert optional_params3["stopSequences"] == ["HALT"]
    assert optional_params3["topP"] == 0.95