from typing import Final

import pytest

import litellm


@pytest.fixture(autouse=True)
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.add_known_models()


LONG_CONTEXT_PROMPT_TOKENS = 300_000
COMPLETION_TOKENS = 1_000

TIERED_COST_CASES = [
    ("gpt-5.4", "flex"),
    ("gpt-5.4-pro", "flex"),
    ("gpt-5.5", "flex"),
    ("gpt-5.6", "priority"),
    ("gpt-5.6-sol", "priority"),
    ("gpt-5.6-terra", "priority"),
    ("gpt-5.6-luna", "priority"),
    ("gpt-6-astra", "priority"),
]


@pytest.mark.parametrize("model,tier", TIERED_COST_CASES)
def test_cost_per_token_bills_long_context_at_the_tier_rate(
    model: str, tier: str
) -> None:
    """A prompt over 272K on flex or priority must bill at that tier's long-context rate."""
    input_cost, output_cost = litellm.cost_per_token(
        model=model,
        prompt_tokens=LONG_CONTEXT_PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        service_tier=tier,
    )
    model_info: Final = litellm.model_cost[model]
    assert input_cost == pytest.approx(
        LONG_CONTEXT_PROMPT_TOKENS * model_info[f"input_cost_per_token_above_272k_tokens_{tier}"]
    )
    assert output_cost == pytest.approx(
        COMPLETION_TOKENS * model_info[f"output_cost_per_token_above_272k_tokens_{tier}"]
    )
