"""
Unit tests for fix #28539: client_metadata should NOT be forwarded from
Responses API bridge to litellm.completion()/acompletion().

The Responses API handler merges **kwargs into the completion args dict.
When a client sends client_metadata (an OpenAI Responses API param), it flows
through kwargs and causes 'unexpected keyword argument' errors at the provider
SDK level. The fix filters kwargs to only pass litellm-internal params
(prefixed with 'litellm_').
"""

import pytest

from litellm.responses.litellm_completion_transformation.handler import (
    _filter_kwargs_for_completion,
)


class TestFilterKwargsForCompletion:
    """Verify _filter_kwargs_for_completion strips non-litellm kwargs."""

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

    def test_strips_all_non_litellm_kwargs(self):
        """Any non-litellm_* kwarg should be stripped."""
        kwargs = {
            "litellm_metadata": {},
            "client_metadata": {"x": 1},
            "some_other_param": True,
            "prompt_id": "p-1",
        }
        filtered = _filter_kwargs_for_completion(kwargs)
        assert filtered == {"litellm_metadata": {}}

    def test_empty_kwargs(self):
        """Empty kwargs should return empty dict."""
        assert _filter_kwargs_for_completion({}) == {}

    def test_no_litellm_kwargs(self):
        """If only non-litellm kwargs, return empty dict."""
        kwargs = {"client_metadata": {}, "extra_param": "value"}
        assert _filter_kwargs_for_completion(kwargs) == {}
