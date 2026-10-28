from typing import Final

import pytest

import litellm
from litellm.cost_calculator import cost_per_token

MODEL: Final = "litellm-test-vertex-character-model"


@pytest.fixture
def character_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        MODEL,
        {
            "litellm_provider": "vertex_ai",
            "mode": "chat",
            "input_cost_per_character": 1e-7,
            "input_cost_per_character_above_128k_tokens": 3e-7,
            "output_cost_per_character": 5e-7,
            "output_cost_per_character_above_128k_tokens": 1e-6,
        },
    )


@pytest.mark.usefixtures("character_row")
def test_completion_characters_bill_at_the_character_price_above_128k_prompt_tokens() -> None:
    """Main billed the completion tokens at the per-character price"""
    _, completion_cost = cost_per_token(
        model=MODEL,
        custom_llm_provider="vertex_ai",
        prompt_tokens=200_000,
        completion_tokens=50_000,
        prompt_characters=0,
        completion_characters=200_000,
    )

    assert completion_cost == pytest.approx(0.2)


@pytest.mark.usefixtures("character_row")
def test_the_character_band_is_picked_by_prompt_tokens_not_characters() -> None:
    """Main picked the band at 32,000 characters (characters * 4 above 128k)"""
    prompt_cost, completion_cost = cost_per_token(
        model=MODEL,
        custom_llm_provider="vertex_ai",
        prompt_tokens=10_000,
        completion_tokens=0,
        prompt_characters=40_000,
        completion_characters=1_000,
    )

    assert (prompt_cost, completion_cost) == (pytest.approx(40_000 * 1e-7), pytest.approx(1_000 * 5e-7))
