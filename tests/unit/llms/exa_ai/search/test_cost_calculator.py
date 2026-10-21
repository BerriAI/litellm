from typing import Final

import pytest

import litellm
from litellm.llms.exa_ai.search.cost_calculator import (
    EXA_AI_ADDITIONAL_RESULT_COST,
    EXA_AI_COST_PARAM,
    EXA_AI_DEFAULT_RESULTS,
    exa_ai_search_cost,
)


@pytest.mark.parametrize("reported_cost", [0.123456789, 0])
def test_exa_search_cost_uses_a_valid_provider_reported_total(reported_cost: float) -> None:
    cost: Final = exa_ai_search_cost("exa_ai/search", {EXA_AI_COST_PARAM: reported_cost})

    assert cost == pytest.approx(reported_cost)


@pytest.mark.parametrize("reported_cost", [True, -0.1, float("nan"), float("inf"), "0.1"])
def test_exa_search_cost_rejects_invalid_provider_totals(reported_cost: object) -> None:
    base_cost: Final = float(litellm.model_cost["exa_ai/search"]["input_cost_per_query"])
    cost: Final = exa_ai_search_cost("exa_ai/search", {EXA_AI_COST_PARAM: reported_cost})

    assert cost == pytest.approx(base_cost)


@pytest.mark.parametrize(
    ("optional_params", "results"),
    [
        ({"max_results": 25}, 25),
        ({"numResults": 25}, 25),
        ({}, EXA_AI_DEFAULT_RESULTS),
    ],
)
def test_exa_search_cost_adds_price_for_results_above_the_included_amount(
    optional_params: dict[str, object], results: int
) -> None:
    base_cost: Final = float(litellm.model_cost["exa_ai/search"]["input_cost_per_query"])
    expected_cost: Final = base_cost + max(results - EXA_AI_DEFAULT_RESULTS, 0) * EXA_AI_ADDITIONAL_RESULT_COST
    cost: Final = exa_ai_search_cost("exa_ai/search", optional_params)

    assert cost == pytest.approx(expected_cost)
