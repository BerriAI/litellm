"""Test Bedrock cross-region inference profile model mapping"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.utils import _get_model_info_helper
from litellm.cost_calculator import completion_cost
from litellm.types.utils import ModelResponse, Usage, Choices, Message


def test_bedrock_cross_region_inference_profile_mapping():
    """Test that bedrock cross-region inference profile model is mapped"""
    model = "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"

    model_info = _get_model_info_helper(model=model, custom_llm_provider="bedrock")

    assert model_info is not None
    assert model_info["litellm_provider"] == "bedrock"
    assert model_info["input_cost_per_token"] == 8e-07


def test_proxy_cost_calculation_scenario():
    """Test exact GitHub issue scenario: proxy cost calculation"""
    model = "litellm_proxy/bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"

    # Test model info lookup works
    model_info = _get_model_info_helper(model=model, custom_llm_provider="litellm_proxy")
    assert model_info is not None

    # Test cost calculation works
    response = ModelResponse(
        id="test",
        created=1234567890,
        model=model,
        object="chat.completion",
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="Test", role="assistant"))],
        usage=Usage(total_tokens=150, prompt_tokens=100, completion_tokens=50),
    )

    cost = completion_cost(completion_response=response, model=model, custom_llm_provider="litellm_proxy")
    expected_cost = (100 * 8e-07) + (50 * 4e-06)
    assert cost == expected_cost