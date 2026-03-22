import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

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


def test_web_search_cost_from_model_info():
    """Cost should come from model_info when search_context_cost_per_query is set."""
    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "search_context_cost_per_query": {
            "search_context_size_low": 0.014,
            "search_context_size_medium": 0.014,
            "search_context_size_high": 0.014,
        },
    }
    cost = cost_per_web_search_request(usage=_make_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.014 * 3)


def test_web_search_cost_legacy_fallback():
    """Without search_context_cost_per_query, should fallback to $0.035."""
    model_info = {"key": "gemini/gemini-2.0-flash"}
    cost = cost_per_web_search_request(usage=_make_usage(2), model_info=model_info)
    assert cost == pytest.approx(0.035 * 2)


def test_web_search_cost_zero_requests():
    """Zero web search requests should return zero cost."""
    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.014,
        },
    }
    cost = cost_per_web_search_request(usage=_make_usage(0), model_info=model_info)
    assert cost == 0.0


def test_web_search_cost_no_usage_details():
    """Missing prompt_tokens_details should return zero cost."""
    model_info = {"key": "gemini/gemini-3-flash-preview"}
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    cost = cost_per_web_search_request(usage=usage, model_info=model_info)
    assert cost == 0.0
