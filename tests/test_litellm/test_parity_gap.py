
import pytest
from litellm.llms.anthropic.chat.transformation import AnthropicConfig

def test_anthropic_response_format_parity_claude_4_7():
    """
    Test that Claude 4.7 correctly maps response_format to native output_format.
    """
    config = AnthropicConfig()
    
    # Test with Claude 4.7 Sonnet - typical model name format
    model = "claude-4-7-sonnet-20261031"
    non_default_params = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object", "properties": {"foo": {"type": "string"}}}
            }
        }
    }
    optional_params = {}
    
    mapped_params = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False
    )
    
    # Verify it uses native output_format
    assert "output_format" in mapped_params, f"Claude 4.7 should use native output_format, but it was not found in {mapped_params.keys()}"
    assert mapped_params["output_format"]["type"] == "json_schema"
    assert "tool_choice" not in mapped_params or mapped_params["tool_choice"]["type"] != "tool", "Should not fall back to tool calling for JSON mode on Claude 4.7"

def test_anthropic_max_completion_tokens_parity():
    """
    Test that max_completion_tokens correctly maps to max_tokens for Anthropic.
    """
    config = AnthropicConfig()
    model = "claude-3-5-sonnet-20240620"
    non_default_params = {"max_completion_tokens": 500}
    optional_params = {}
    
    mapped_params = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False
    )
    
    assert mapped_params["max_tokens"] == 500

def test_bedrock_max_completion_tokens_parity():
    """
    Test that max_completion_tokens correctly maps to maxTokens for Bedrock.
    """
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    config = AmazonConverseConfig()
    model = "anthropic.claude-3-5-sonnet-v1:0"
    non_default_params = {"max_completion_tokens": 750}
    optional_params = {}
    
    mapped_params = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False
    )
    
    assert mapped_params["maxTokens"] == 750
