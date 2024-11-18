# What is this?
## Unit testing for the 'get_model_info()' function
import os
import sys
import traceback

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
        litellm.get_model_info(model, custom_llm_provider=provider)
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
    from litellm.llms.ollama import OllamaConfig

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

        assert info["supports_function_calling"] is True

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        assert mock_client.call_args.kwargs["json"]["name"] == "mistral"
