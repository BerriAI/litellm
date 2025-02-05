import json
import os
import sys
import copy
import re
import pytest

# Adjust the system path to include the parent directory
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.utils import get_llm_provider, ProviderConfigManager
from litellm import LlmProviders


@pytest.mark.asyncio
async def test_supports_tool_choice():
    # Path to the JSON file
    json_path = "../../model_prices_and_context_window.json"

    # Load the JSON data (used for logic, not for output)
    with open(json_path, "r", encoding="utf-8") as f:
        model_prices = json.load(f)

    # Create a deep copy to prevent accidental modifications to the original data structure
    model_prices_copy = copy.deepcopy(model_prices)
    litellm.model_cost = model_prices_copy

    config_manager = ProviderConfigManager()

    # List to keep track of models that need to be updated
    models_to_update = []

    # Define a block list of models that should not be updated
    block_list = {
        "bedrock/ap-northeast-1/1-month-commitment/anthropic.claude-v1",
        "bedrock/ap-northeast-1/6-month-commitment/anthropic.claude-v1",
        "bedrock/eu-central-1/1-month-commitment/anthropic.claude-v1",
        "bedrock/eu-central-1/6-month-commitment/anthropic.claude-v1",
        "bedrock/us-east-1/1-month-commitment/anthropic.claude-v1",
        "bedrock/us-east-1/6-month-commitment/anthropic.claude-v1",
        "bedrock/us-west-2/1-month-commitment/anthropic.claude-v1",
        "bedrock/us-west-2/6-month-commitment/anthropic.claude-v1",
        "bedrock/ap-northeast-1/1-month-commitment/anthropic.claude-v2",
        "bedrock/ap-northeast-1/6-month-commitment/anthropic.claude-v2",
        "bedrock/eu-central-1/1-month-commitment/anthropic.claude-v2",
        "bedrock/eu-central-1/6-month-commitment/anthropic.claude-v2",
        "bedrock/us-east-1/1-month-commitment/anthropic.claude-v2",
        "bedrock/us-east-1/6-month-commitment/anthropic.claude-v2",
        "bedrock/us-west-2/1-month-commitment/anthropic.claude-v2",
        "bedrock/us-west-2/6-month-commitment/anthropic.claude-v2",
        "bedrock/ap-northeast-1/1-month-commitment/anthropic.claude-v2:1",
        "bedrock/ap-northeast-1/6-month-commitment/anthropic.claude-v2:1",
        "bedrock/eu-central-1/1-month-commitment/anthropic.claude-v2:1",
        "bedrock/eu-central-1/6-month-commitment/anthropic.claude-v2:1",
        "bedrock/us-east-1/1-month-commitment/anthropic.claude-v2:1",
        "bedrock/us-east-1/6-month-commitment/anthropic.claude-v2:1",
        "bedrock/us-west-2/1-month-commitment/anthropic.claude-v2:1",
        "bedrock/us-west-2/6-month-commitment/anthropic.claude-v2:1",
        "bedrock/us-east-1/1-month-commitment/anthropic.claude-instant-v1",
        "bedrock/us-east-1/6-month-commitment/anthropic.claude-instant-v1",
        "bedrock/us-west-2/1-month-commitment/anthropic.claude-instant-v1",
        "bedrock/us-west-2/6-month-commitment/anthropic.claude-instant-v1",
        "bedrock/ap-northeast-1/1-month-commitment/anthropic.claude-instant-v1",
        "bedrock/ap-northeast-1/6-month-commitment/anthropic.claude-instant-v1",
        "bedrock/eu-central-1/1-month-commitment/anthropic.claude-instant-v1",
        "bedrock/eu-central-1/6-month-commitment/anthropic.claude-instant-v1",
        "bedrock/*/1-month-commitment/cohere.command-text-v14",
        "bedrock/*/6-month-commitment/cohere.command-text-v14",
        "bedrock/*/1-month-commitment/cohere.command-light-text-v14",
        "bedrock/*/6-month-commitment/cohere.command-light-text-v14",
    }

    for model_name, model_info in model_prices.items():
        # Skip irrelevant models
        if model_name == "sample_spec":
            continue
        if model_info.get("mode") != "chat":
            continue
        if model_info.get("litellm_provider") == "aleph_alpha":
            continue
        _litellm_provider = model_info.get("litellm_provider")
        if "azure/mistral" in model_name or "azure/command-r" in model_name:
            continue

        # Check if the model is in the block list
        if model_name in block_list:
            continue

        # Attempt to get the LLM provider
        try:
            model, provider, _, _ = get_llm_provider(model=model_name)
        except Exception as e:
            if _litellm_provider == "bedrock":
                try:
                    model, provider, _, _ = get_llm_provider(
                        model=model_name, custom_llm_provider="bedrock"
                    )
                except Exception as e:
                    try:
                        model, provider, _, _ = get_llm_provider(
                            model=model_name, custom_llm_provider="bedrock_converse"
                        )
                    except Exception as e:
                        print(
                            f"\033[91mERROR for {model_name} with bedrock_converse: {e}\033[0m"
                        )
                    continue
            else:
                print(f"\033[91mERROR for {model_name}: {e}\033[0m")
                continue

        # Get provider config
        provider_enum = LlmProviders(provider)
        config = config_manager.get_provider_chat_config(model, provider_enum)

        # Get supported params
        supported_params = config.get_supported_openai_params(model)
        if provider == "bedrock":
            print("provider=", provider)
            print("model=", model)
            print("supported_params=", supported_params)
            print("\n")

        # Check if 'tool_choice' is supported and not yet added in the model info
        has_tool_choice = "tool_choice" in supported_params

        if has_tool_choice and "supports_tool_choice" not in model_info:
            models_to_update.append(model_name)

    if models_to_update:
        # Instead of re-dumping the JSON (which re-formats other fields),
        # load the file contents as text and update only the "tool_choice" property.

        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Iterate over each model that needs updating. The regex below assumes
        # that the JSON file is pretty printed (with newlines and indents).
        for model_name in models_to_update:
            # This pattern matches the block for a model such as:
            # "model_name": {
            #     "key1": value1,
            #     "key2": value2
            # }
            # It captures the opening of the object, the inner content, and the closing newline/indent.
            pattern = re.compile(
                r'("' + re.escape(model_name) + r'"\s*:\s*{)(.*?)(\n\s*})', re.DOTALL
            )

            def replacer(match):
                before = match.group(1)
                inner = match.group(2)
                after = match.group(3)
                # If the 'tool_choice' key is already present for some reason, do nothing.
                if '"tool_choice"' in inner:
                    return match.group(0)
                # Determine the indent for inner object keys (based on the closing brace line)
                indent_match = re.search(r"\n(\s*)\}", after)
                indent_space = indent_match.group(1) if indent_match else ""
                inner_str = inner.rstrip()
                # If there are other keys, add a comma if one is not already there;
                # otherwise, simply insert the new key/value with proper indent.
                if inner_str:
                    if inner_str[-1] != ",":
                        new_inner = (
                            inner
                            + ",\n"
                            + indent_space
                            + '    "supports_tool_choice": true'
                        )
                    else:
                        new_inner = (
                            inner
                            + "\n"
                            + indent_space
                            + '    "supports_tool_choice": true'
                        )
                else:
                    new_inner = (
                        "\n" + indent_space + '    "supports_tool_choice": true\n'
                    )
                return before + new_inner + after

            # Update the file content only for the current model.
            content, count = pattern.subn(replacer, content)
            if count:
                print(f"Added 'tool_choice' to model: {model_name}")

        # Write the (minimally) modified content back to the file.
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(content)
