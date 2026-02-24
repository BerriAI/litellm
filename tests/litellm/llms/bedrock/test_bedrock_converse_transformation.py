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
