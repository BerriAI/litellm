from unittest.mock import patch

from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

MODEL_INFO = {
    "input_cost_per_token": 4e-07,
    "output_cost_per_token": 1.6e-06,
    "cache_read_input_token_cost": 8e-08,
}


def _usage(prompt_tokens, cached_tokens, completion_tokens=100):
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )


@patch("litellm.llms.fireworks_ai.cost_calculator.get_model_info", return_value=MODEL_INFO)
def test_cached_prompt_tokens_use_cache_read_cost(mock_info):
    """Cached prefix tokens should be billed at cache_read_input_token_cost."""
    prompt_cost, _ = cost_per_token(model="qwen", usage=_usage(10000, 9000))
    expected = 1000 * MODEL_INFO["input_cost_per_token"] + 9000 * MODEL_INFO[
        "cache_read_input_token_cost"
    ]
    assert prompt_cost == expected


@patch("litellm.llms.fireworks_ai.cost_calculator.get_model_info", return_value=MODEL_INFO)
def test_no_cached_tokens_charges_full_price(mock_info):
    """With no cached tokens, every prompt token is billed at the input rate."""
    prompt_cost, _ = cost_per_token(model="qwen", usage=_usage(10000, 0))
    assert prompt_cost == 10000 * MODEL_INFO["input_cost_per_token"]
