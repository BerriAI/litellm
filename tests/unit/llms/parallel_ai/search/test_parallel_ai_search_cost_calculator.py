import pytest

import litellm
from litellm.llms.parallel_ai.search.cost_calculator import parallel_ai_search_cost
from litellm.utils import (
    _invalidate_model_cost_lowercase_map,  # pyright: ignore[reportPrivateUsage]  # setitem bypasses the lowercase index
)


def test_results_past_the_included_ten_are_priced_from_the_bundled_row():
    row = litellm.model_cost["parallel_ai/search"]
    query_rate, result_rate = row["input_cost_per_query"], row["cost_per_additional_result"]
    assert isinstance(query_rate, float) and isinstance(result_rate, float)
    expected = query_rate + 15 * result_rate
    assert parallel_ai_search_cost({"max_results": 25}, None) == pytest.approx(expected)


@pytest.mark.parametrize(("cost_per_additional_result", "expected"), [(None, 0.005), (0.002, 0.005 + 15 * 0.002)])
def test_a_registered_row_prices_extra_results_from_its_own_field(
    monkeypatch: pytest.MonkeyPatch, cost_per_additional_result: float | None, expected: float
):
    row = {"litellm_provider": "parallel_ai", "mode": "search", "input_cost_per_query": 0.005}
    if cost_per_additional_result is not None:
        row["cost_per_additional_result"] = cost_per_additional_result
    monkeypatch.setitem(litellm.model_cost, "parallel_ai/search", row)
    _invalidate_model_cost_lowercase_map()  # pyright: ignore[reportPrivateUsage]  # setitem bypasses the lowercase index

    assert parallel_ai_search_cost({"max_results": 25}, None) == pytest.approx(expected)
