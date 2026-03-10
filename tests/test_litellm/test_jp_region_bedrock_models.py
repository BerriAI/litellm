"""
Test that jp. (Japan) region prefix exists for Bedrock models in model_prices_and_context_window.json
https://github.com/BerriAI/litellm/issues/22972
"""

import json
import os

import litellm


def test_jp_anthropic_claude_sonnet_4_6_exists():
    """
    Test that jp.anthropic.claude-sonnet-4-6 exists in model_prices_and_context_window.json
    with correct configuration matching other regional prefixes (us, eu, au).
    """
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    assert "jp.anthropic.claude-sonnet-4-6" in model_data, \
        "Missing Japan region model: jp.anthropic.claude-sonnet-4-6"

    info = model_data["jp.anthropic.claude-sonnet-4-6"]

    # Verify provider and basic config
    assert info["litellm_provider"] == "bedrock_converse"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 200000
    assert info["max_output_tokens"] == 64000
    assert info["max_tokens"] == 64000

    # Verify capabilities
    assert info["supports_assistant_prefill"] is True
    assert info["supports_computer_use"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_pdf_input"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["tool_use_system_prompt_tokens"] == 346


def test_jp_claude_sonnet_4_6_matches_other_regions():
    """
    Test that jp.anthropic.claude-sonnet-4-6 has the same pricing and config
    as au.anthropic.claude-sonnet-4-6 (both are non-global regional prefixes).
    """
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    jp_info = model_data["jp.anthropic.claude-sonnet-4-6"]
    au_info = model_data["au.anthropic.claude-sonnet-4-6"]

    keys_to_match = [
        "litellm_provider",
        "max_input_tokens",
        "max_output_tokens",
        "max_tokens",
        "mode",
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_token_above_200k_tokens",
        "output_cost_per_token_above_200k_tokens",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_200k_tokens",
        "cache_read_input_token_cost",
        "cache_read_input_token_cost_above_200k_tokens",
        "tool_use_system_prompt_tokens",
    ]

    for key in keys_to_match:
        assert jp_info[key] == au_info[key], \
            f"Mismatch for {key}: jp={jp_info[key]}, au={au_info[key]}"


def test_jp_claude_sonnet_4_6_registered_in_bedrock_converse(monkeypatch):
    """Test that jp.anthropic.claude-sonnet-4-6 is registered in bedrock_converse_models."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    import importlib
    importlib.reload(litellm)
    assert "jp.anthropic.claude-sonnet-4-6" in litellm.bedrock_converse_models, \
        "jp.anthropic.claude-sonnet-4-6 not registered in bedrock_converse_models"


def test_jp_amazon_nova_2_lite_exists():
    """
    Test that jp.amazon.nova-2-lite-v1:0 exists in model_prices_and_context_window.json
    with correct configuration matching apac.amazon.nova-2-lite-v1:0.
    """
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    assert "jp.amazon.nova-2-lite-v1:0" in model_data, \
        "Missing Japan region model: jp.amazon.nova-2-lite-v1:0"

    info = model_data["jp.amazon.nova-2-lite-v1:0"]

    # Verify provider and basic config
    assert info["litellm_provider"] == "bedrock_converse"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 64000
    assert info["max_tokens"] == 64000

    # Verify capabilities
    assert info["supports_function_calling"] is True
    assert info["supports_pdf_input"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_video_input"] is True
    assert info["supports_vision"] is True


def test_jp_amazon_nova_2_lite_matches_apac():
    """
    Test that jp.amazon.nova-2-lite-v1:0 has the same pricing and config
    as apac.amazon.nova-2-lite-v1:0.
    """
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    jp_info = model_data["jp.amazon.nova-2-lite-v1:0"]
    apac_info = model_data["apac.amazon.nova-2-lite-v1:0"]

    keys_to_match = [
        "litellm_provider",
        "max_input_tokens",
        "max_output_tokens",
        "max_tokens",
        "mode",
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
    ]

    for key in keys_to_match:
        assert jp_info[key] == apac_info[key], \
            f"Mismatch for {key}: jp={jp_info[key]}, apac={apac_info[key]}"


def test_jp_amazon_nova_2_lite_registered_in_bedrock_converse(monkeypatch):
    """Test that jp.amazon.nova-2-lite-v1:0 is registered in bedrock_converse_models."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    import importlib
    importlib.reload(litellm)
    assert "jp.amazon.nova-2-lite-v1:0" in litellm.bedrock_converse_models, \
        "jp.amazon.nova-2-lite-v1:0 not registered in bedrock_converse_models"
