"""
Tests that ``get_cost_for_anthropic_web_search`` tolerates ``server_tool_use``
being either a ``dict`` or a ``ServerToolUse`` pydantic instance.

See https://github.com/BerriAI/litellm/issues/26153.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.anthropic.cost_calculation import (
    _get_web_search_requests,
    get_cost_for_anthropic_web_search,
)
from litellm.types.utils import ModelInfo, ServerToolUse


class _UsageWithServerToolUse:
    def __init__(self, server_tool_use):
        self.server_tool_use = server_tool_use


def _make_model_info(cost_per_query: float = 0.01) -> ModelInfo:
    info: ModelInfo = {  # type: ignore[typeddict-item]
        "search_context_cost_per_query": {
            "search_context_size_low": cost_per_query,
            "search_context_size_medium": cost_per_query,
            "search_context_size_high": cost_per_query,
        }
    }
    return info


def test_get_web_search_requests_handles_none():
    assert _get_web_search_requests(None) is None


def test_get_web_search_requests_handles_dict():
    assert _get_web_search_requests({"web_search_requests": 4}) == 4


def test_get_web_search_requests_handles_dict_missing_key():
    assert _get_web_search_requests({}) is None


def test_get_web_search_requests_handles_pydantic():
    assert _get_web_search_requests(ServerToolUse(web_search_requests=2)) == 2


def test_get_cost_for_anthropic_web_search_with_dict_server_tool_use():
    """
    Regression: ``server_tool_use`` was a dict from ``stream_chunk_builder`` and
    direct attribute access on it raised ``AttributeError``.
    """
    usage = _UsageWithServerToolUse({"web_search_requests": 3})
    info = _make_model_info(cost_per_query=0.01)

    cost = get_cost_for_anthropic_web_search(
        model_info=info, usage=usage  # type: ignore[arg-type]
    )

    assert cost == pytest.approx(0.03)


def test_get_cost_for_anthropic_web_search_with_pydantic_server_tool_use():
    usage = _UsageWithServerToolUse(ServerToolUse(web_search_requests=3))
    info = _make_model_info(cost_per_query=0.01)

    cost = get_cost_for_anthropic_web_search(
        model_info=info, usage=usage  # type: ignore[arg-type]
    )

    assert cost == pytest.approx(0.03)


def test_get_cost_for_anthropic_web_search_with_none_server_tool_use():
    usage = _UsageWithServerToolUse(None)
    info = _make_model_info(cost_per_query=0.01)

    cost = get_cost_for_anthropic_web_search(
        model_info=info, usage=usage  # type: ignore[arg-type]
    )

    assert cost == 0.0


def test_get_cost_for_anthropic_web_search_with_no_usage():
    info = _make_model_info(cost_per_query=0.01)
    cost = get_cost_for_anthropic_web_search(model_info=info, usage=None)
    assert cost == 0.0
