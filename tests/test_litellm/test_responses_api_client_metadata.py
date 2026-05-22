"""
Unit tests for fix #28539: client_metadata should NOT be forwarded from
Responses API bridge to litellm.completion()/acompletion().

The Responses API handler merges **kwargs into the completion args dict.
When a client sends client_metadata (an OpenAI Responses API param), it flows
through kwargs and causes 'unexpected keyword argument' errors at the provider
SDK level. The fix filters kwargs via a blocklist of Responses-API-only params,
preserving valid completion kwargs like api_key, mock_response, and litellm_*.
"""

import pytest

from litellm.responses.litellm_completion_transformation.handler import (
    _filter_kwargs_for_completion,
)


class TestFilterKwargsForCompletion:
    """Verify _filter_kwargs_for_completion strips Responses-API-only kwargs."""

    def test_strips_client_metadata(self):
        """client_metadata from the Responses API must not pass through."""
        kwargs = {
            "litellm_logging_obj": "mock_obj",
            "litellm_metadata": {"key": "value"},
            "client_metadata": {"session": "abc123"},
        }
        filtered = _filter_kwargs_for_completion(kwargs)
        assert "client_metadata" not in filtered
        assert filtered["litellm_logging_obj"] == "mock_obj"
        assert filtered["litellm_metadata"] == {"key": "value"}

    def test_preserves_all_litellm_prefixed_kwargs(self):
        """All litellm_* kwargs should be preserved."""
        kwargs = {
            "litellm_call_id": "call-123",
            "litellm_metadata": {},
            "litellm_logging_obj": None,
            "litellm_parent_otel_span": "span",
        }
        filtered = _filter_kwargs_for_completion(kwargs)
        assert filtered == kwargs

    def test_preserves_api_key(self):
        """api_key must pass through for BYOK proxy support."""
        kwargs = {
            "api_key": "sk-user-provided-key",
            "litellm_metadata": {},
            "client_metadata": {"x": 1},
        }
        filtered = _filter_kwargs_for_completion(kwargs)
        assert filtered["api_key"] == "sk-user-provided-key"
        assert "client_metadata" not in filtered

    def test_preserves_valid_completion_kwargs(self):
        """Valid litellm.completion() kwargs like mock_response, num_retries, drop_params
        must pass through (they don't have a litellm_ prefix)."""
        kwargs = {
            "mock_response": "hello",
            "num_retries": 3,
            "drop_params": True,
            "input_cost_per_token": 0.01,
            "client_metadata": {"x": 1},
        }
        filtered = _filter_kwargs_for_completion(kwargs)
        assert filtered["mock_response"] == "hello"
        assert filtered["num_retries"] == 3
        assert filtered["drop_params"] is True
        assert filtered["input_cost_per_token"] == 0.01
        assert "client_metadata" not in filtered

    def test_strips_responses_api_only_params(self):
        """Responses-API-only params should be stripped."""
        kwargs = {
            "litellm_metadata": {},
            "client_metadata": {"x": 1},
            "include": ["file_search_results"],
            "instructions": "Be helpful",
            "previous_response_id": "resp_abc",
            "truncation": "auto",
        }
        filtered = _filter_kwargs_for_completion(kwargs)
        assert filtered == {"litellm_metadata": {}}

    def test_empty_kwargs(self):
        """Empty kwargs should return empty dict."""
        assert _filter_kwargs_for_completion({}) == {}
