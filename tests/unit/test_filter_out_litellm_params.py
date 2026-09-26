"""
Test filter_out_litellm_params helper function.
"""

import pytest

import litellm
import litellm.types.utils as types_utils
from litellm.utils import filter_out_litellm_params


def test_filter_out_litellm_params():
    """
    Test that filter_out_litellm_params removes LiteLLM internal parameters
    while keeping provider-specific parameters.
    """
    kwargs = {
        "query": "test query",
        "max_results": 10,
        "shared_session": "mock_session_object",
        "metadata": {"key": "value"},
        "litellm_trace_id": "trace-123",
        "proxy_server_request": {"url": "http://example.com"},
        "secret_fields": {"api_key": "secret"},
        "custom_param": "should_be_kept",
    }

    filtered = filter_out_litellm_params(kwargs=kwargs)

    # Provider-specific params are kept
    assert filtered["query"] == "test query"
    assert filtered["max_results"] == 10
    assert filtered["custom_param"] == "should_be_kept"

    # LiteLLM internal params are removed
    assert "shared_session" not in filtered
    assert "metadata" not in filtered
    assert "litellm_trace_id" not in filtered
    assert "proxy_server_request" not in filtered
    assert "secret_fields" not in filtered


def test_filter_out_litellm_params_also_drops_the_excluded_names():
    kwargs = {"temperature": 0.2, "top_k": 5, "litellm_trace_id": "trace-1", "_litellm_control": object()}

    assert filter_out_litellm_params(kwargs, excluding=("temperature",)) == {"top_k": 5}


def test_filter_out_litellm_params_sees_a_name_appended_to_the_public_list_after_import():
    litellm.all_litellm_params.append("registered_later")
    try:
        filtered = filter_out_litellm_params({"registered_later": 1, "top_k": 2})
    finally:
        litellm.all_litellm_params.remove("registered_later")

    assert filtered == {"top_k": 2}


def test_filter_out_litellm_params_sees_the_public_list_rebound_after_import(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(types_utils, "all_litellm_params", (*types_utils.all_litellm_params, "registered_later"))

    assert filter_out_litellm_params({"registered_later": 1, "top_k": 2}) == {"top_k": 2}
