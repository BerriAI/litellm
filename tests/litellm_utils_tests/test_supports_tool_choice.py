import json
import os
import sys
from unittest.mock import patch
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.utils import get_llm_provider
from litellm.utils import ProviderConfigManager
from litellm import LlmProviders

OLD_PROVIDERS = ["aleph_alpha", "palm"]


@pytest.mark.asyncio
async def test_supports_tool_choice():
    """
    goes through all models in model_prices_and_context_window.json, checks if the litellm.utils.supports_tool_choice() returns the value set in the model_prices_and_context_window.json
    """
    # Load the model prices file
    with open("../../model_prices_and_context_window.json", "r") as f:
        model_prices = json.load(f)
    litellm.model_cost = model_prices

    config_manager = ProviderConfigManager()

    for model_name in model_prices:
        # Get LLM provider
        print(f"testing model: {model_name}")
        if model_name == "sample_spec":
            continue
        model_info = model_prices[model_name]

        if model_info.get("mode", None) != "chat":
            continue
        _litellm_provider = model_info["litellm_provider"]
        if "azure/mistral" in model_name or "azure/command-r" in model_name:
            continue
        if any(provider in model_name for provider in OLD_PROVIDERS):
            continue
        if _litellm_provider in OLD_PROVIDERS:
            continue

        try:
            model, provider, _, _ = get_llm_provider(model=model_name)
        except Exception as e:
            print(f"\033[91mERROR for {model_name}: {e}\033[0m")
            continue

        # Get provider config
        provider_enum = LlmProviders(provider)
        config = config_manager.get_provider_chat_config(model, provider_enum)

        # Get supported params
        supported_params = config.get_supported_openai_params(model)

        # Check if tool_choice is in supported params
        has_tool_choice = litellm.utils.supports_tool_choice(
            model=model_name, custom_llm_provider=provider
        )

        # Print in red if there's a mismatch between supported_params and has_tool_choice
        if ("tool_choice" in supported_params) != has_tool_choice:
            print(
                f"\033[91mINCORRECT VALUE for {model_name}: has_tool_choice={has_tool_choice}, tool_choice in supported_params={'tool_choice' in supported_params}\033[0m"
            )
        else:
            print(
                f"CORRECT: {model_name} - has_tool_choice={has_tool_choice}, tool_choice in supported_params={'tool_choice' in supported_params}"
            )
