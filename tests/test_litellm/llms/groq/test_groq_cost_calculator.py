import pytest

from litellm.llms.groq.cost_calculator import cost_per_web_search_request
from litellm.types.utils import ModelInfo, ServerToolUse, Usage

PRICED_MODEL_INFO = ModelInfo(
    key="groq/openai/gpt-oss-20b",
    litellm_provider="groq",
    mode="chat",
    search_context_cost_per_query={
        "search_context_size_low": 0.005,
        "search_context_size_medium": 0.005,
        "search_context_size_high": 0.005,
    },
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
    unpriced = ModelInfo(key="groq/openai/gpt-oss-20b", litellm_provider="groq", mode="chat")
    cost = cost_per_web_search_request(usage=_usage_with_actions(3), model_info=unpriced)
    assert cost == 0.0


def test_bills_visit_website_per_open():
    cost = cost_per_web_search_request(usage=_usage_with_actions(0, opens=15), model_info=PRICED_MODEL_INFO)
    assert cost == pytest.approx(15 * 0.001)


def test_bills_searches_and_opens_together():
    cost = cost_per_web_search_request(usage=_usage_with_actions(2, opens=3), model_info=PRICED_MODEL_INFO)
    assert cost == pytest.approx(2 * 0.005 + 3 * 0.001)
