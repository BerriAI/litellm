"""
Test Claude Haiku 4.5 model configurations for Bedrock
https://github.com/BerriAI/litellm/issues/15818
"""

import json
import os


def test_bedrock_haiku_4_5_configuration():
    """Test that all Bedrock Claude Haiku 4.5 models use bedrock_converse provider"""
    # Load model configuration
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    # All Bedrock Haiku 4.5 variants that should use bedrock_converse
    bedrock_haiku_models = [
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-haiku-4-5@20251001",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "apac.anthropic.claude-haiku-4-5-20251001-v1:0",
        "jp.anthropic.claude-haiku-4-5-20251001-v1:0",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "au.anthropic.claude-haiku-4-5-20251001-v1:0",
    ]

    for model in bedrock_haiku_models:
        assert model in model_data, f"Model {model} not found in config"
        model_info = model_data[model]

        # Verify uses bedrock_converse (not legacy bedrock provider)
        assert (
            model_info["litellm_provider"] == "bedrock_converse"
        ), f"{model} should use bedrock_converse provider, got {model_info['litellm_provider']}"

        # Verify supports vision (key missing capability)
        assert model_info.get("supports_vision") is True, f"{model} should support vision"

        # Verify tool use system prompt tokens
        assert (
            model_info.get("tool_use_system_prompt_tokens") == 346
        ), f"{model} should have tool_use_system_prompt_tokens set to 346"

        # Verify core capabilities
        assert model_info.get("supports_computer_use") is True
        assert model_info.get("supports_function_calling") is True
        assert model_info.get("supports_tool_choice") is True
        assert model_info.get("supports_prompt_caching") is True
        assert model_info.get("supports_response_schema") is True
        assert model_info.get("supports_pdf_input") is True
        assert model_info.get("supports_assistant_prefill") is True
        assert model_info.get("supports_reasoning") is True

        # Verify token limits
        assert model_info["max_input_tokens"] == 200000
        assert model_info["max_output_tokens"] == 64000
        assert model_info["mode"] == "chat"


def test_bedrock_haiku_4_5_matches_sonnet_capabilities():
    """
    Test that Haiku 4.5 has same capabilities as Sonnet 4.5
    (including computer_use, vision, tools, etc.)
    """
    # Load model configuration
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    haiku_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    sonnet_model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    haiku_info = model_data[haiku_model]
    sonnet_info = model_data[sonnet_model]

    # Both should use bedrock_converse
    assert haiku_info["litellm_provider"] == "bedrock_converse"
    assert sonnet_info["litellm_provider"] == "bedrock_converse"

    # Shared capabilities that should match
    shared_capabilities = [
        "supports_vision",
        "supports_computer_use",
        "supports_function_calling",
        "supports_tool_choice",
        "supports_prompt_caching",
        "supports_response_schema",
        "supports_pdf_input",
        "supports_assistant_prefill",
        "supports_reasoning",
        "tool_use_system_prompt_tokens",
    ]

    for capability in shared_capabilities:
        assert haiku_info.get(capability) == sonnet_info.get(
            capability
        ), f"Capability {capability} mismatch: Haiku={haiku_info.get(capability)}, Sonnet={sonnet_info.get(capability)}"


def test_anthropic_api_haiku_4_5_configuration():
    """Test that Anthropic API Claude Haiku 4.5 has correct configuration"""
    # Load model configuration
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    # Anthropic API models (not Bedrock)
    anthropic_models = [
        "claude-haiku-4-5-20251001",
        "claude-haiku-4-5",
    ]

    for model in anthropic_models:
        assert model in model_data, f"Model {model} not found in config"
        model_info = model_data[model]

        # Should use anthropic provider (not bedrock)
        assert model_info["litellm_provider"] == "anthropic", f"{model} should use anthropic provider"

        # Should support vision
        assert model_info.get("supports_vision") is True, f"{model} should support vision"

        # Should have larger output token limit (64K for Anthropic API)
        assert model_info["max_output_tokens"] == 64000
