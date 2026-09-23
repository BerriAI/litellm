from typing import Final

import pytest

import litellm
from litellm.llms.vertex_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


@pytest.mark.parametrize(
    ("text_tokens", "expected_prompt_cost"),
    [
        (140_000, 140_000 * 0.002 + 120_000 * 0.0005),
        (120_000, 120_000 * 0.001 + 120_000 * 0.0005),
    ],
    ids=["creation_tokens_do_not_change_the_tier_rate", "creation_tokens_cannot_push_the_tier_threshold"],
)
def test_above_128k_pricing_splits_cache_creation_tokens_out_of_the_prompt(
    monkeypatch: pytest.MonkeyPatch, text_tokens: int, expected_prompt_cost: float
) -> None:
    """Cache creation tokens bill at the cache-creation rate and never count toward the above-128k tier."""
    model: Final = "vertex_ai/fake-above-128k-model"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "vertex_ai",
            "input_cost_per_token": 0.001,
            "input_cost_per_token_above_128k_tokens": 0.002,
            "cache_creation_input_token_cost": 0.0005,
            "output_cost_per_token": 0.003,
        },
    )

    usage: Final = Usage(
        prompt_tokens=text_tokens + 120_000,
        completion_tokens=10,
        total_tokens=text_tokens + 120_010,
        prompt_tokens_details=PromptTokensDetailsWrapper(cache_creation_tokens=120_000),
    )
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        custom_llm_provider="vertex_ai",
        usage=usage,
    )

    assert prompt_cost == pytest.approx(expected_prompt_cost)
    assert completion_cost == pytest.approx(10 * 0.003)
