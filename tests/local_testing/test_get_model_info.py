# What is this?
## Unit testing for the 'get_model_info()' function
import os
import sys
import traceback

from typing import List, Dict, Any

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm import get_model_info
from unittest.mock import AsyncMock, MagicMock, patch


def test_get_model_info_simple_model_name():
    """
    tests if model name given, and model exists in model info - the object is returned
    """
    model = "claude-3-opus-20240229"
    litellm.get_model_info(model)


def test_get_model_info_custom_llm_with_model_name():
    """
    Tests if {custom_llm_provider}/{model_name} name given, and model exists in model info, the object is returned
    """
    model = "anthropic/claude-3-opus-20240229"
    litellm.get_model_info(model)


def test_get_model_info_custom_llm_with_same_name_vllm():
    """
    Tests if {custom_llm_provider}/{model_name} name given, and model exists in model info, the object is returned
    """
    model = "command-r-plus"
    provider = "openai"  # vllm is openai-compatible
    try:
        model_info = litellm.get_model_info(model, custom_llm_provider=provider)
        print("model_info", model_info)
        pytest.fail("Expected get model info to fail for an unmapped model/provider")
    except Exception:
        pass


def test_get_model_info_shows_correct_supports_vision():
    info = litellm.get_model_info("gemini/gemini-1.5-flash")
    print("info", info)
    assert info["supports_vision"] is True


def test_get_model_info_shows_assistant_prefill():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    info = litellm.get_model_info("deepseek/deepseek-chat")
    print("info", info)
    assert info.get("supports_assistant_prefill") is True


def test_get_model_info_shows_supports_prompt_caching():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    info = litellm.get_model_info("deepseek/deepseek-chat")
    print("info", info)
    assert info.get("supports_prompt_caching") is True


def test_get_model_info_finetuned_models():
    info = litellm.get_model_info("ft:gpt-3.5-turbo:my-org:custom_suffix:id")
    print("info", info)
    assert info["input_cost_per_token"] == 0.000003


def test_get_model_info_gemini_pro():
    info = litellm.get_model_info("gemini-1.5-pro-002")
    print("info", info)
    assert info["key"] == "gemini-1.5-pro-002"


def test_get_model_info_ollama_chat():
    from litellm.llms.ollama.completion.transformation import OllamaConfig

    with patch.object(
        litellm.module_level_client,
        "post",
        return_value=MagicMock(
            json=lambda: {
                "model_info": {"llama.context_length": 32768},
                "template": "tools",
            }
        ),
    ) as mock_client:
        info = OllamaConfig().get_model_info("mistral")
        assert info["supports_function_calling"] is True

        info = get_model_info("ollama/mistral")
        print("info", info)
        assert info["supports_function_calling"] is True

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        assert mock_client.call_args.kwargs["json"]["name"] == "mistral"


def test_get_model_info_gemini():
    """
    Tests if ALL gemini models have 'tpm' and 'rpm' in the model info
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_map = litellm.model_cost
    for model, info in model_map.items():
        if model.startswith("gemini/") and not "gemma" in model:
            assert info.get("tpm") is not None, f"{model} does not have tpm"
            assert info.get("rpm") is not None, f"{model} does not have rpm"


def test_get_model_info_bedrock_region():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    args = {
        "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "custom_llm_provider": "bedrock",
    }
    litellm.model_cost.pop("us.anthropic.claude-3-5-sonnet-20241022-v2:0", None)
    info = litellm.get_model_info(**args)
    print("info", info)
    assert info["key"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert info["litellm_provider"] == "bedrock"


@pytest.mark.parametrize(
    "model",
    [
        "ft:gpt-3.5-turbo:my-org:custom_suffix:id",
        "ft:gpt-4-0613:my-org:custom_suffix:id",
        "ft:davinci-002:my-org:custom_suffix:id",
        "ft:gpt-4-0613:my-org:custom_suffix:id",
        "ft:babbage-002:my-org:custom_suffix:id",
        "gpt-35-turbo",
        "ada",
    ],
)
def test_get_model_info_completion_cost_unit_tests(model):
    info = litellm.get_model_info(model)
    print("info", info)


def test_get_model_info_ft_model_with_provider_prefix():
    args = {
        "model": "openai/ft:gpt-3.5-turbo:my-org:custom_suffix:id",
        "custom_llm_provider": "openai",
    }
    info = litellm.get_model_info(**args)
    print("info", info)
    assert info["key"] == "ft:gpt-3.5-turbo"


def test_get_whitelisted_models():
    """
    Snapshot of all bedrock models as of 12/24/2024.

    Enforce any new bedrock chat model to be added as `bedrock_converse` unless explicitly whitelisted.

    Create whitelist to prevent naming regressions for older litellm versions.
    """
    whitelisted_models = []
    for model, info in litellm.model_cost.items():
        if info["litellm_provider"] == "bedrock" and info["mode"] == "chat":
            whitelisted_models.append(model)

        # Write to a local file
    with open("whitelisted_bedrock_models.txt", "w") as file:
        for model in whitelisted_models:
            file.write(f"{model}\n")

    print("whitelisted_models written to whitelisted_bedrock_models.txt")


def _enforce_bedrock_converse_models(
    model_cost: List[Dict[str, Any]], whitelist_models: List[str]
):
    """
    Assert all new bedrock chat models are added as `bedrock_converse` unless explicitly whitelisted.
    """
    # Check for unwhitelisted models
    for model, info in litellm.model_cost.items():
        if (
            info["litellm_provider"] == "bedrock"
            and info["mode"] == "chat"
            and model not in whitelist_models
        ):
            raise AssertionError(
                f"New bedrock chat model detected: {model}. Please set `litellm_provider='bedrock_converse'` for this model."
            )


def test_model_info_bedrock_converse(monkeypatch):
    """
    Assert all new bedrock chat models are added as `bedrock_converse` unless explicitly whitelisted.

    This ensures they are automatically routed to the converse endpoint.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Load whitelist models from file
    with open("whitelisted_bedrock_models.txt", "r") as file:
        whitelist_models = [line.strip() for line in file.readlines()]

    _enforce_bedrock_converse_models(
        model_cost=litellm.model_cost, whitelist_models=whitelist_models
    )


def test_model_info_bedrock_converse_enforcement(monkeypatch):
    """
    Test the enforcement of the whitelist by adding a fake model and ensuring the test fails.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Add a fake unwhitelisted model
    litellm.model_cost["fake.bedrock-chat-model"] = {
        "litellm_provider": "bedrock",
        "mode": "chat",
    }

    # Load whitelist models from file
    with open("whitelisted_bedrock_models.txt", "r") as file:
        whitelist_models = [line.strip() for line in file.readlines()]

    # Check for unwhitelisted models
    with pytest.raises(AssertionError):
        _enforce_bedrock_converse_models(
            model_cost=litellm.model_cost, whitelist_models=whitelist_models
        )
