"""
Validate Claude Opus 4.6 model configuration entries.
"""

import json
import os

import litellm


def test_claude_4_6_australia_region_uses_au_prefix_not_apac():
    """
    Test that Australia region Claude 4.6 models use 'au.' prefix instead of incorrect 'apac.' prefix.

    AWS Bedrock cross-region inference uses specific regional prefixes:
    - 'us.' for United States
    - 'eu.' for Europe
    - 'au.' for Australia (ap-southeast-2)
    - 'apac.' for Asia-Pacific (Singapore, ap-southeast-1)

    This test ensures the Claude 4.6 models correctly use 'au.' for Australia,
    and that 'apac.' is NOT incorrectly used for Australia region.

    Related: The 'apac.' prefix is valid for Asia-Pacific (Singapore) region models,
    but should not be used for Australia which has its own 'au.' prefix.
    """
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        model_data = json.load(f)

    # Verify au.anthropic.claude-opus-4-6-v1 exists (correct)
    assert (
        "au.anthropic.claude-opus-4-6-v1" in model_data
    ), "Missing Australia region model: au.anthropic.claude-opus-4-6-v1"

    # Verify apac.anthropic.claude-opus-4-6-v1 does NOT exist (incorrect)
    assert (
        "apac.anthropic.claude-opus-4-6-v1" not in model_data
    ), "Incorrect model entry exists: apac.anthropic.claude-opus-4-6-v1 should be au.anthropic.claude-opus-4-6-v1"

    # Verify au.anthropic.claude-sonnet-4-6 exists (correct)
    assert (
        "au.anthropic.claude-sonnet-4-6" in model_data
    ), "Missing Australia region model: au.anthropic.claude-sonnet-4-6"

    # Verify apac.anthropic.claude-sonnet-4-6 does NOT exist (incorrect)
    assert (
        "apac.anthropic.claude-sonnet-4-6" not in model_data
    ), "Incorrect model entry exists: apac.anthropic.claude-sonnet-4-6 should be au.anthropic.claude-sonnet-4-6"

    # Verify the au. model is registered in bedrock_converse_models
    assert (
        "au.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models
    ), "au.anthropic.claude-opus-4-6-v1 not registered in bedrock_converse_models"

    # Verify apac. is NOT registered for this model
    assert (
        "apac.anthropic.claude-opus-4-6-v1" not in litellm.bedrock_converse_models
    ), "apac.anthropic.claude-opus-4-6-v1 should not be in bedrock_converse_models"

    # Verify the au. model is registered in bedrock_converse_models
    assert (
        "au.anthropic.claude-sonnet-4-6" in litellm.bedrock_converse_models
    ), "au.anthropic.claude-sonnet-4-6 not registered in bedrock_converse_models"

    # Verify apac. is NOT registered for this model
    assert (
        "apac.anthropic.claude-sonnet-4-6" not in litellm.bedrock_converse_models
    ), "apac.anthropic.claude-sonnet-4-6 should not be in bedrock_converse_models"


def test_opus_4_6_alias_and_dated_metadata_match():
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        model_data = json.load(f)

    alias = model_data["claude-opus-4-6"]
    dated = model_data["claude-opus-4-6-20260205"]

    keys_to_match = [
        "max_input_tokens",
        "max_output_tokens",
        "max_tokens",
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
        "cache_read_input_token_cost",
        "supports_assistant_prefill",
    ]
    for key in keys_to_match:
        assert alias[key] == dated[key], f"Mismatch for {key}"


def test_opus_4_6_bedrock_converse_registration():
    assert "anthropic.claude-opus-4-6-v1" in litellm.BEDROCK_CONVERSE_MODELS
    assert "global.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models
    assert "us.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models
    assert "eu.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models
    assert "au.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models
