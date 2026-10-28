from typing import cast

import pytest

from litellm.llms.groq.cost_calculator import cost_per_web_search_request
from litellm.types.utils import ModelInfo, ServerToolUse, Usage


def _model_info(**fields: object) -> ModelInfo:
    return cast(ModelInfo, fields)


PRICED_MODEL_INFO = _model_info(
    key="groq/openai/gpt-oss-20b",
    litellm_provider="groq",
    mode="chat",
    search_context_cost_per_query={
        "search_context_size_low": 0.005,
        "search_context_size_medium": 0.005,
        "search_context_size_high": 0.005,
    },
    browser_open_cost_per_request=0.001,
)


def _usage_with_actions(searches: int | None, opens: int | None = None) -> Usage:
    return Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        server_tool_use=ServerToolUse(web_search_requests=searches, browser_open_requests=opens),
    )


def test_bills_per_executed_search():
    cost = cost_per_web_search_request(usage=_usage_with_actions(7), model_info=PRICED_MODEL_INFO)
    assert cost == pytest.approx(7 * 0.005)


def test_no_searches_costs_nothing():
    cost = cost_per_web_search_request(usage=_usage_with_actions(None), model_info=PRICED_MODEL_INFO)
    assert cost == 0.0


def test_missing_usage_signal_costs_nothing():
    usage = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    cost = cost_per_web_search_request(usage=usage, model_info=PRICED_MODEL_INFO)
    assert cost == 0.0


def test_missing_pricing_costs_nothing():
    unpriced = _model_info(key="groq/openai/gpt-oss-20b", litellm_provider="groq", mode="chat")
    cost = cost_per_web_search_request(usage=_usage_with_actions(3), model_info=unpriced)
    assert cost == 0.0


def test_bills_visit_website_per_open():
    cost = cost_per_web_search_request(usage=_usage_with_actions(0, opens=15), model_info=PRICED_MODEL_INFO)
    assert cost == pytest.approx(15 * 0.001)


def test_bills_searches_and_opens_together():
    cost = cost_per_web_search_request(usage=_usage_with_actions(2, opens=3), model_info=PRICED_MODEL_INFO)
    assert cost == pytest.approx(2 * 0.005 + 3 * 0.001)


@pytest.mark.parametrize(("browser_open_cost_per_request", "expected"), [(None, 0.0), (0.002, 3 * 0.002)])
def test_browser_opens_are_priced_from_the_row(browser_open_cost_per_request: float | None, expected: float):
    model_info = _model_info(
        key="groq/my-registered-model",
        litellm_provider="groq",
        mode="chat",
        browser_open_cost_per_request=browser_open_cost_per_request,
    )
    cost = cost_per_web_search_request(usage=_usage_with_actions(0, opens=3), model_info=model_info)
    assert cost == pytest.approx(expected)


def test_every_bundled_groq_browser_row_prices_opens():
    from litellm._catalog import CATALOG

    rows = CATALOG.bundled().served()
    searchable = [
        key
        for key, row in rows.items()
        if row.get("litellm_provider") == "groq" and "search_context_cost_per_query" in row
    ]
    assert searchable
    assert all(rows[key].get("browser_open_cost_per_request") for key in searchable)
