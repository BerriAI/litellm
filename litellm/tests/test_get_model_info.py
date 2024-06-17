# What is this?
## Unit testing for the 'get_model_info()' function
import os, sys, traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import get_model_info
import pytest


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
