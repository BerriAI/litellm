import pytest

from litellm.llms.gemini.cost_calculator import cost_per_web_search_request
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


def _make_usage(web_search_requests: int) -> Usage:
    return Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            web_search_requests=web_search_requests,
        ),
    )


def test_per_query_billing():
    """web_search_billing_unit=per_query charges per search query."""
    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.014,
        },
    }
    cost = cost_per_web_search_request(usage=_make_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.014 * 3)


def test_per_prompt_billing():
    """web_search_billing_unit=per_prompt (default) clamps to 1."""
    model_info = {
        "key": "gemini/gemini-2.5-flash",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.035,
        },
    }
    cost = cost_per_web_search_request(usage=_make_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.035 * 1)


def test_default_billing_unit_is_per_prompt():
    """Without web_search_billing_unit, defaults to per_prompt (clamp to 1)."""
    model_info = {"key": "gemini/gemini-2.0-flash"}
    cost = cost_per_web_search_request(usage=_make_usage(2), model_info=model_info)
    assert cost == pytest.approx(0.035 * 1)


def test_zero_requests():
    """Zero web search requests should return zero cost."""
    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "web_search_billing_unit": "per_query",
    }
    cost = cost_per_web_search_request(usage=_make_usage(0), model_info=model_info)
    assert cost == 0.0


def test_no_usage_details():
    """Missing prompt_tokens_details should return zero cost."""
    model_info = {"key": "gemini/gemini-3-flash-preview"}
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    cost = cost_per_web_search_request(usage=usage, model_info=model_info)
    assert cost == 0.0
