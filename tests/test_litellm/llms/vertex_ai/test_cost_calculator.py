from typing import Final

import pytest

import litellm
from litellm.llms.vertex_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


@pytest.mark.parametrize(
    ("text_tokens", "cache_read_tokens", "cache_creation_tokens", "expected_prompt_cost"),
    [
        (140_000, 0, 120_000, 140_000 * 0.002 + 120_000 * 0.0005),
        (120_000, 0, 120_000, 120_000 * 0.001 + 120_000 * 0.0005),
        (100_000, 50_000, 0, 100_000 * 0.002 + 50_000 * 0.00025),
        (100_000, 0, 120_000, 100_000 * 0.001 + 120_000 * 0.0005),
    ],
    ids=[
        "creation_tokens_do_not_change_the_tier_rate",
        "creation_tokens_cannot_push_the_tier_threshold",
        "cache_read_tokens_count_toward_the_tier",
        "small_text_plus_creation_stays_below_the_tier",
    ],
)
def test_above_128k_pricing_splits_cache_tokens_out_of_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
    text_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    expected_prompt_cost: float,
) -> None:
    """Cache reads count toward the above-128k tier, cache creation bills at the cache-creation rate."""
    model: Final = "vertex_ai/fake-above-128k-model"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "vertex_ai",
            "input_cost_per_token": 0.001,
            "input_cost_per_token_above_128k_tokens": 0.002,
            "cache_read_input_token_cost": 0.00025,
            "cache_creation_input_token_cost": 0.0005,
            "output_cost_per_token": 0.003,
        },
    )

    usage: Final = Usage(
        prompt_tokens=text_tokens + cache_read_tokens + cache_creation_tokens,
        completion_tokens=10,
        total_tokens=text_tokens + cache_read_tokens + cache_creation_tokens + 10,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cache_read_tokens or 0,
            cache_creation_tokens=cache_creation_tokens or 0,
        ),
    )
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        custom_llm_provider="vertex_ai",
        usage=usage,
    )

    assert prompt_cost == pytest.approx(expected_prompt_cost)
    assert completion_cost == pytest.approx(10 * 0.003)


def test_above_128k_pricing_falls_back_to_the_resolved_tier_rate_for_creation_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing cache-creation rate resolves to the tiered input rate, like generic_cost_per_token."""
    model: Final = "vertex_ai/fake-above-128k-model-no-creation-rate"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "vertex_ai",
            "input_cost_per_token": 0.001,
            "input_cost_per_token_above_128k_tokens": 0.002,
            "output_cost_per_token": 0.003,
        },
    )

    usage: Final = Usage(
        prompt_tokens=260_000,
        completion_tokens=10,
        total_tokens=260_010,
        prompt_tokens_details=PromptTokensDetailsWrapper(cache_creation_tokens=120_000),
    )
    prompt_cost, _ = cost_per_token(
        model=model,
        custom_llm_provider="vertex_ai",
        usage=usage,
    )

    assert prompt_cost == pytest.approx(140_000 * 0.002 + 120_000 * 0.002)
