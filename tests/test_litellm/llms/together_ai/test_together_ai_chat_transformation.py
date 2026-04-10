import os
import sys

# Add litellm to path
sys.path.insert(0, os.path.abspath("../../../.."))
import litellm
from litellm.llms.together_ai.chat import TogetherAIConfig


def _use_local_model_cost_map() -> None:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")


def test_together_ai_gpt_oss_supports_reasoning_effort():
    """
    gpt-oss models on together_ai support reasoning. Regression test for #25132.
    """
    _use_local_model_cost_map()

    supported_params = TogetherAIConfig().get_supported_openai_params(
        model="together_ai/openai/gpt-oss-120b"
    )
    assert "reasoning_effort" in supported_params

    supported_params_20b = TogetherAIConfig().get_supported_openai_params(
        model="together_ai/openai/gpt-oss-20b"
    )
    assert "reasoning_effort" in supported_params_20b


def test_together_ai_non_reasoning_model_does_not_expose_reasoning_effort():
    """
    Non-reasoning together_ai models should not advertise reasoning_effort.
    """
    _use_local_model_cost_map()

    supported_params = TogetherAIConfig().get_supported_openai_params(
        model="together_ai/meta-llama/Llama-3-70b-chat-hf"
    )
    assert "reasoning_effort" not in supported_params


def test_together_ai_gpt_oss_reasoning_effort_flows_through_get_optional_params():
    """
    End-to-end: the exact call from issue #25132 must not raise
    UnsupportedParamsError, and reasoning_effort must land in optional_params.
    """
    _use_local_model_cost_map()

    optional_params = litellm.get_optional_params(
        model="openai/gpt-oss-120b",
        custom_llm_provider="together_ai",
        reasoning_effort="low",
    )
    assert optional_params.get("reasoning_effort") == "low"
