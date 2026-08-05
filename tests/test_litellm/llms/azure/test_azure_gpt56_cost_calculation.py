import pytest

import litellm
from litellm.types.utils import Choices, Message, ModelResponse, Usage


def test_completion_cost_supports_luna_dated_snapshot(monkeypatch):
    """Azure response model snapshots should resolve to canonical pricing."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    response = ModelResponse(
        model="gpt-5.6-luna-2026-07-09",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    response._hidden_params = {"custom_llm_provider": "azure"}

    assert "azure/gpt-5.6-luna-2026-07-09" in litellm.model_cost
    assert litellm.completion_cost(completion_response=response) == pytest.approx(0.0004)
