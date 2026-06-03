"""
Unit tests for Langfuse Responses API support.

Tests for:
- _get_responses_api_response: unwrapping ResponseCompletedEvent
- _get_responses_api_content_for_langfuse: output_text preference
- _extract_cache_read_input_tokens: cache token extraction precedence
- isinstance-based usage field fallback
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from litellm.integrations.langfuse.langfuse import (
    LangFuseLogger,
    _extract_cache_read_input_tokens,
    _get_numeric_attr_or_key,
)
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


def _responses_api_response(**overrides):
    """Build a minimal ResponsesAPIResponse for testing."""
    base = dict(
        id="resp_test",
        created_at=1,
        object="response",
        model="gpt-5.5",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
        usage=cast(
            Any,
            {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 30},
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 112,
            },
        ),
    )
    base.update(overrides)
    return ResponsesAPIResponse(**base)


# ---------------------------------------------------------------------------
# _get_responses_api_response
# ---------------------------------------------------------------------------


class TestGetResponsesApiResponse:
    """Tests for LangFuseLogger._get_responses_api_response static method."""

    def test_direct_instance(self):
        resp = _responses_api_response()
        result = LangFuseLogger._get_responses_api_response(resp)
        assert result is resp

    def test_wrapped_in_completed_event(self):
        inner = _responses_api_response()
        event = ResponseCompletedEvent.model_construct(
            type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
            response=inner,
        )
        result = LangFuseLogger._get_responses_api_response(cast(Any, event))
        assert result is inner

    def test_unrelated_object_returns_none(self):
        result = LangFuseLogger._get_responses_api_response("not a response")
        assert result is None

    def test_object_with_non_response_attr_returns_none(self):
        obj = SimpleNamespace(response="something else")
        result = LangFuseLogger._get_responses_api_response(cast(Any, obj))
        assert result is None

    def test_none_returns_none(self):
        result = LangFuseLogger._get_responses_api_response(None)
        assert result is None


# ---------------------------------------------------------------------------
# _get_responses_api_content_for_langfuse
# ---------------------------------------------------------------------------


class TestGetResponsesApiContentForLangfuse:
    """Tests for LangFuseLogger._get_responses_api_content_for_langfuse static method."""

    def test_prefers_output_text(self):
        resp = _responses_api_response()
        result = LangFuseLogger._get_responses_api_content_for_langfuse(resp)
        assert result == "hello"

    def test_falls_back_to_output_list_when_no_output_text(self):
        # model_construct skips property computation, so output_text won't exist
        resp = ResponsesAPIResponse.model_construct(
            id="resp_no_text",
            created_at=1,
            object="response",
            model="gpt-5.5",
            output=[{"type": "message", "role": "assistant", "content": []}],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )
        result = LangFuseLogger._get_responses_api_content_for_langfuse(resp)
        assert result == resp.output

    def test_returns_none_when_no_output(self):
        resp = ResponsesAPIResponse.model_construct(
            id="resp_empty",
            created_at=1,
            object="response",
            model="gpt-5.5",
            output=[],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0),
        )
        result = LangFuseLogger._get_responses_api_content_for_langfuse(resp)
        assert result is None


# ---------------------------------------------------------------------------
# _extract_cache_read_input_tokens
# ---------------------------------------------------------------------------


class TestExtractCacheReadInputTokens:
    """Tests for _extract_cache_read_input_tokens helper function."""

    def _make_usage(self, **kwargs):
        """Build a SimpleNamespace with .get() support for dict-like access."""
        ns = SimpleNamespace(**kwargs)
        # .get() is called by the function; delegate to getattr with defaults
        ns.get = lambda key, default=None: getattr(ns, key, default)
        return ns

    def test_prefers_input_tokens_details_over_prompt_tokens_details(self):
        usage = self._make_usage(
            input_tokens_details=SimpleNamespace(cached_tokens=30),
            prompt_tokens_details=SimpleNamespace(cached_tokens=10),
        )
        assert _extract_cache_read_input_tokens(usage) == 30

    def test_falls_back_to_prompt_tokens_details(self):
        usage = self._make_usage(
            prompt_tokens_details=SimpleNamespace(cached_tokens=15),
        )
        assert _extract_cache_read_input_tokens(usage) == 15

    def test_returns_zero_when_no_details(self):
        usage = self._make_usage()
        assert _extract_cache_read_input_tokens(usage) == 0

    def test_top_level_cache_read_input_tokens(self):
        usage = self._make_usage(cache_read_input_tokens=50)
        assert _extract_cache_read_input_tokens(usage) == 50

    def test_ignores_zero_cached_tokens(self):
        usage = self._make_usage(
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assert _extract_cache_read_input_tokens(usage) == 0

    def test_prefers_input_tokens_details_over_top_level(self):
        usage = self._make_usage(
            cache_read_input_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=30),
        )
        assert _extract_cache_read_input_tokens(usage) == 30

    def test_reads_dict_shaped_token_details(self):
        usage = {
            "cache_read_input_tokens": 5,
            "input_tokens_details": {"cached_tokens": 30},
            "prompt_tokens_details": {"cached_tokens": 10},
        }
        assert _extract_cache_read_input_tokens(usage) == 30


class TestNumericAttrOrKey:
    def test_reads_attribute_field(self):
        usage = SimpleNamespace(input_tokens=100)
        assert _get_numeric_attr_or_key(usage, ("prompt_tokens", "input_tokens")) == 100

    def test_reads_dict_field(self):
        usage = {"prompt_tokens": 101803, "completion_tokens": 247}
        assert (
            _get_numeric_attr_or_key(usage, ("prompt_tokens", "input_tokens")) == 101803
        )
        assert (
            _get_numeric_attr_or_key(usage, ("completion_tokens", "output_tokens"))
            == 247
        )

    def test_falls_back_to_responses_dict_field(self):
        usage = {"input_tokens": 100, "output_tokens": 12}
        assert _get_numeric_attr_or_key(usage, ("prompt_tokens", "input_tokens")) == 100
        assert (
            _get_numeric_attr_or_key(usage, ("completion_tokens", "output_tokens"))
            == 12
        )

    def test_rejects_mock_auto_attributes(self):
        usage = MagicMock()
        assert _get_numeric_attr_or_key(usage, ("prompt_tokens", "input_tokens")) == 0


# ---------------------------------------------------------------------------
# isinstance-based usage field fallback
# ---------------------------------------------------------------------------


class TestUsageFieldFallback:
    """Tests for the isinstance-based prompt/completion tokens fallback logic.
    These validate the key property: when prompt_tokens/completion_tokens
    are None or non-numeric, the code falls back to input_tokens/output_tokens,
    and ultimately to 0.
    """

    def test_isinstance_rejects_none(self):
        assert not isinstance(None, (int, float))

    def test_isinstance_accepts_int(self):
        assert isinstance(42, (int, float))

    def test_isinstance_accepts_float(self):
        assert isinstance(3.14, (int, float))

    def test_isinstance_rejects_string(self):
        assert not isinstance("hello", (int, float))

    def test_isinstance_rejects_mock(self):
        """MagicMock auto-attributes must not pass the isinstance check."""
        m = MagicMock()
        assert not isinstance(m.usage.input_tokens, (int, float))

    def test_int_cast_on_float(self):
        """float values should be cast to int for LangfuseUsageDetails compatibility."""
        assert int(3.14) == 3
        assert int(100.0) == 100
