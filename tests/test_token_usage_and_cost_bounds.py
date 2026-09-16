import pytest
import litellm
from litellm import completion_cost, Usage

def test_usage_dataclass_initialization():
    usage = Usage(prompt_tokens=150, completion_tokens=50, total_tokens=200)
    assert usage.prompt_tokens == 150
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 200

def test_completion_cost_with_zero_tokens():
    mock_resp = {
        "model": "gpt-3.5-turbo",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    cost = completion_cost(completion_response=mock_resp)
    assert cost == 0.0

def test_model_cost_lookup():
    cost_map = litellm.model_cost
    assert isinstance(cost_map, dict)
    assert "gpt-4" in cost_map or "gpt-3.5-turbo" in cost_map