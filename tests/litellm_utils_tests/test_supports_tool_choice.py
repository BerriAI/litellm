import json
import os
import sys
from unittest.mock import patch
import pytest

# Add parent directory to system path
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.utils import get_llm_provider, ProviderConfigManager, _check_provider_match
from litellm import LlmProviders


def test_supports_tool_choice_simple_tests():
    """
    simple sanity checks
    """
    assert litellm.utils.supports_tool_choice(model="gpt-4o") == True
    assert (
        litellm.utils.supports_tool_choice(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        )
        == True
    )
    assert (
        litellm.utils.supports_tool_choice(
            model="anthropic.claude-3-sonnet-20240229-v1:0"
        )
        is True
    )

    assert (
        litellm.utils.supports_tool_choice(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            custom_llm_provider="bedrock_converse",
        )
        is True
    )

    assert (
        litellm.utils.supports_tool_choice(model="us.amazon.nova-micro-v1:0") is False
    )
    assert (
        litellm.utils.supports_tool_choice(model="bedrock/us.amazon.nova-micro-v1:0")
        is False
    )
    assert (
        litellm.utils.supports_tool_choice(
            model="us.amazon.nova-micro-v1:0", custom_llm_provider="bedrock_converse"
        )
        is False
    )


def test_check_provider_match():
    """
    Test the _check_provider_match function for various provider scenarios
    """
    # Test bedrock and bedrock_converse cases
    model_info = {"litellm_provider": "bedrock"}
    assert litellm.utils._check_provider_match(model_info, "bedrock") is True
    assert litellm.utils._check_provider_match(model_info, "bedrock_converse") is True

    # Test bedrock_converse provider
    model_info = {"litellm_provider": "bedrock_converse"}
    assert litellm.utils._check_provider_match(model_info, "bedrock") is True
    assert litellm.utils._check_provider_match(model_info, "bedrock_converse") is True

    # Test non-matching provider
    model_info = {"litellm_provider": "bedrock"}
    assert litellm.utils._check_provider_match(model_info, "openai") is False


# Models that should be skipped during testing
OLD_PROVIDERS = ["aleph_alpha", "palm"]
SKIP_MODELS = ["azure/mistral", "azure/command-r", "jamba", "deepinfra", "mistral."]

# Bedrock models to block - organized by type
BEDROCK_REGIONS = ["ap-northeast-1", "eu-central-1", "us-east-1", "us-west-2"]
BEDROCK_COMMITMENTS = ["1-month-commitment", "6-month-commitment"]
BEDROCK_MODELS = {
    "anthropic.claude-v1",
    "anthropic.claude-v2",
    "anthropic.claude-v2:1",
    "anthropic.claude-instant-v1",
}

# Generate block_list dynamically
block_list = set()
for region in BEDROCK_REGIONS:
    for commitment in BEDROCK_COMMITMENTS:
        for model in BEDROCK_MODELS:
            block_list.add(f"bedrock/{region}/{commitment}/{model}")
            block_list.add(f"bedrock/{region}/{model}")

# Add Cohere models
for commitment in BEDROCK_COMMITMENTS:
    block_list.add(f"bedrock/*/{commitment}/cohere.command-text-v14")
    block_list.add(f"bedrock/*/{commitment}/cohere.command-light-text-v14")

print("block_list", block_list)


@pytest.mark.asyncio
async def test_supports_tool_choice():
    """
    Test that litellm.utils.supports_tool_choice() returns the correct value
    for all models in model_prices_and_context_window.json.

    The test:
    1. Loads model pricing data
    2. Iterates through each model
    3. Checks if tool_choice support matches the model's supported parameters
    """
    # Load model prices
    litellm._turn_on_debug()
    with open("./model_prices_and_context_window.json", "r") as f:
        model_prices = json.load(f)
    litellm.model_cost = model_prices
    config_manager = ProviderConfigManager()

    for model_name, model_info in model_prices.items():
        print(f"testing model: {model_name}")

        # Skip certain models
        if (
            model_name == "sample_spec"
            or model_info.get("mode") != "chat"
            or any(skip in model_name for skip in SKIP_MODELS)
            or any(provider in model_name for provider in OLD_PROVIDERS)
            or model_info["litellm_provider"] in OLD_PROVIDERS
            or model_name in block_list
        ):
            continue

        try:
            model, provider, _, _ = get_llm_provider(model=model_name)
        except Exception as e:
            print(f"\033[91mERROR for {model_name}: {e}\033[0m")
            continue

        # Get provider config and supported params
        print("LLM provider", provider)
        provider_enum = LlmProviders(provider)
        config = config_manager.get_provider_chat_config(model, provider_enum)
        supported_params = config.get_supported_openai_params(model)
        print("supported_params", supported_params)

        # Check tool_choice support
        supports_tool_choice_result = litellm.utils.supports_tool_choice(
            model=model_name, custom_llm_provider=provider
        )
        tool_choice_in_params = "tool_choice" in supported_params

        assert supports_tool_choice_result == tool_choice_in_params, (
            f"Tool choice support mismatch for {model_name}:\n"
            f"supports_tool_choice() returned: {supports_tool_choice_result}\n"
            f"tool_choice in supported params: {tool_choice_in_params}"
        )
