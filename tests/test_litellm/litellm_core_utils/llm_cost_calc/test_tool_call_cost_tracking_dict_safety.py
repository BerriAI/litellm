"""
Tests that the cost-tracking call sites tolerate ``server_tool_use`` being
either a ``dict`` or a ``ServerToolUse`` pydantic instance.

See https://github.com/BerriAI/litellm/issues/26153.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
    _get_web_search_requests,
)
from litellm.types.utils import ModelResponse, ServerToolUse, Usage


class _UsageWithDictServerToolUse:
    """
    Tiny stand-in that mimics the broken streaming-rebuild shape:
    ``server_tool_use`` is a plain dict.
    """

    def __init__(self, server_tool_use):
        self.server_tool_use = server_tool_use
        self.prompt_tokens_details = None


def test_get_web_search_requests_handles_none():
    assert _get_web_search_requests(None) is None


def test_get_web_search_requests_handles_dict():
    assert _get_web_search_requests({"web_search_requests": 5}) == 5


def test_get_web_search_requests_handles_dict_missing_key():
    assert _get_web_search_requests({}) is None


def test_get_web_search_requests_handles_pydantic():
    stu = ServerToolUse(web_search_requests=7)
    assert _get_web_search_requests(stu) == 7


def test_get_web_search_requests_handles_pydantic_with_none_value():
    stu = ServerToolUse()
    assert _get_web_search_requests(stu) is None


def test_response_object_includes_web_search_call_with_dict_server_tool_use():
    """
    The exact bug: ``usage.server_tool_use`` is a dict and the check in
    ``response_object_includes_web_search_call`` used to crash with
    ``AttributeError``.
    """
    response = ModelResponse()
    usage = _UsageWithDictServerToolUse({"web_search_requests": 2})

    # Must not raise — and must correctly detect the web search call.
    result = StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
        response_object=response, usage=usage  # type: ignore[arg-type]
    )
    assert result is True


def test_response_object_includes_web_search_call_with_pydantic_server_tool_use():
    response = ModelResponse()
    usage = _UsageWithDictServerToolUse(ServerToolUse(web_search_requests=2))

    result = StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
        response_object=response, usage=usage  # type: ignore[arg-type]
    )
    assert result is True


def test_response_object_includes_web_search_call_with_none_server_tool_use():
    response = ModelResponse()
    usage = _UsageWithDictServerToolUse(None)

    result = StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
        response_object=response, usage=usage  # type: ignore[arg-type]
    )
    assert result is False
