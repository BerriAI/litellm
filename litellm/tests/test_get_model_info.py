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
