"""
Unit tests for passthrough guardrails functionality.

Tests the opt-in guardrail execution model for passthrough endpoints:
- Guardrails only run when explicitly configured
- Field-level targeting with JSONPath expressions
- Automatic inheritance from org/team/key levels when enabled
"""

import pytest

from litellm.proxy._types import PassThroughGuardrailSettings
from litellm.proxy.pass_through_endpoints.jsonpath_extractor import JsonPathExtractor
from litellm.proxy.pass_through_endpoints.passthrough_guardrails import (
    PassthroughGuardrailHandler,
)


class TestPassthroughGuardrailHandlerIsEnabled:
    """Tests for PassthroughGuardrailHandler.is_enabled method."""

    def test_returns_false_when_config_is_none(self):
        """Guardrails should be disabled when config is None."""
        result = PassthroughGuardrailHandler.is_enabled(None)
        assert result is False

    def test_returns_false_when_config_is_empty_dict(self):
        """Guardrails should be disabled when config is empty dict."""
        result = PassthroughGuardrailHandler.is_enabled({})
        assert result is False

    def test_returns_true_when_config_is_list(self):
        """Guardrails should be enabled when config is a list of names."""
        result = PassthroughGuardrailHandler.is_enabled(["pii-detection"])
        assert result is True

    def test_returns_false_when_config_is_invalid_type(self):
        """Guardrails should be disabled when config is not a dict or list."""
        result = PassthroughGuardrailHandler.is_enabled("pii-detection")  # type: ignore
        assert result is False

    def test_returns_true_when_config_has_guardrails(self):
        """Guardrails should be enabled when config has at least one guardrail."""
        config = {"pii-detection": None}
        result = PassthroughGuardrailHandler.is_enabled(config)
        assert result is True

    def test_returns_true_with_multiple_guardrails(self):
        """Guardrails should be enabled with multiple guardrails configured."""
        config = {
            "pii-detection": None,
            "content-moderation": {"request_fields": ["input"]},
        }
        result = PassthroughGuardrailHandler.is_enabled(config)
        assert result is True


class TestPassthroughGuardrailHandlerGetGuardrailNames:
    """Tests for PassthroughGuardrailHandler.get_guardrail_names method."""

    def test_returns_empty_list_when_disabled(self):
        """Should return empty list when guardrails are disabled."""
        result = PassthroughGuardrailHandler.get_guardrail_names(None)
        assert result == []

    def test_returns_guardrail_names(self):
        """Should return list of guardrail names from config."""
        config = {
            "pii-detection": None,
            "content-moderation": {"request_fields": ["input"]},
        }
        result = PassthroughGuardrailHandler.get_guardrail_names(config)
        assert set(result) == {"pii-detection", "content-moderation"}


class TestPassthroughGuardrailHandlerNormalizeConfig:
    """Tests for PassthroughGuardrailHandler.normalize_config method."""

    def test_normalizes_list_to_dict(self):
        """List of guardrail names should be converted to dict with None values."""
        config = ["pii-detection", "content-moderation"]
        result = PassthroughGuardrailHandler.normalize_config(config)
        assert result == {"pii-detection": None, "content-moderation": None}

    def test_returns_dict_unchanged(self):
        """Dict config should be returned as-is."""
        config = {"pii-detection": {"request_fields": ["query"]}}
        result = PassthroughGuardrailHandler.normalize_config(config)
        assert result == config


class TestPassthroughGuardrailHandlerGetSettings:
    """Tests for PassthroughGuardrailHandler.get_settings method."""

    def test_returns_none_when_config_is_none(self):
        """Should return None when config is None."""
        result = PassthroughGuardrailHandler.get_settings(None, "pii-detection")
        assert result is None

    def test_returns_none_when_guardrail_not_in_config(self):
        """Should return None when guardrail is not in config."""
        config = {"pii-detection": None}
        result = PassthroughGuardrailHandler.get_settings(config, "content-moderation")
        assert result is None

    def test_returns_none_when_settings_is_none(self):
        """Should return None when guardrail has no settings."""
        config = {"pii-detection": None}
        result = PassthroughGuardrailHandler.get_settings(config, "pii-detection")
        assert result is None

    def test_returns_settings_object(self):
        """Should return PassThroughGuardrailSettings when settings are provided."""
        config = {
            "pii-detection": {
                "request_fields": ["input", "query"],
                "response_fields": ["output"],
            }
        }
        result = PassthroughGuardrailHandler.get_settings(config, "pii-detection")
        assert result is not None
        assert result.request_fields == ["input", "query"]
        assert result.response_fields == ["output"]


class TestJsonPathExtractorEvaluate:
    """Tests for JsonPathExtractor.evaluate method."""

    def test_simple_key(self):
        """Should extract simple key from dict."""
        data = {"query": "test query", "other": "value"}
        result = JsonPathExtractor.evaluate(data, "query")
        assert result == "test query"

    def test_nested_key(self):
        """Should extract nested key from dict."""
        data = {"foo": {"bar": "nested value"}}
        result = JsonPathExtractor.evaluate(data, "foo.bar")
        assert result == "nested value"

    def test_array_wildcard(self):
        """Should extract values from array using wildcard."""
        data = {"items": [{"text": "item1"}, {"text": "item2"}, {"text": "item3"}]}
        result = JsonPathExtractor.evaluate(data, "items[*].text")
        assert result == ["item1", "item2", "item3"]

    def test_messages_content(self):
        """Should extract content from messages array."""
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        }
        result = JsonPathExtractor.evaluate(data, "messages[*].content")
        assert result == ["You are helpful", "Hello"]

    def test_missing_key_returns_none(self):
        """Should return None for missing key."""
        data = {"query": "test"}
        result = JsonPathExtractor.evaluate(data, "missing")
        assert result is None

    def test_empty_data_returns_none(self):
        """Should return None for empty data."""
        result = JsonPathExtractor.evaluate({}, "query")
        assert result is None

    def test_empty_expression_returns_none(self):
        """Should return None for empty expression."""
        data = {"query": "test"}
        result = JsonPathExtractor.evaluate(data, "")
        assert result is None


class TestJsonPathExtractorExtractFields:
    """Tests for JsonPathExtractor.extract_fields method."""

    def test_extracts_multiple_fields(self):
        """Should extract and concatenate multiple fields."""
        data = {"query": "search query", "input": "additional input"}
        result = JsonPathExtractor.extract_fields(data, ["query", "input"])
        assert "search query" in result
        assert "additional input" in result

    def test_extracts_array_fields(self):
        """Should extract and concatenate array fields."""
        data = {"documents": [{"text": "doc1"}, {"text": "doc2"}]}
        result = JsonPathExtractor.extract_fields(data, ["documents[*].text"])
        assert "doc1" in result
        assert "doc2" in result

    def test_handles_missing_fields(self):
        """Should handle missing fields gracefully."""
        data = {"query": "test"}
        result = JsonPathExtractor.extract_fields(data, ["query", "missing"])
        assert result == "test"

    def test_empty_fields_returns_empty_string(self):
        """Should return empty string for empty fields list."""
        data = {"query": "test"}
        result = JsonPathExtractor.extract_fields(data, [])
        assert result == ""


class TestPassthroughGuardrailHandlerPrepareInput:
    """Tests for PassthroughGuardrailHandler.prepare_input method."""

    def test_returns_full_payload_when_no_settings(self):
        """Should return full JSON payload when no settings provided."""
        data = {"query": "test", "input": "value"}
        result = PassthroughGuardrailHandler.prepare_input(data, None)
        assert "query" in result
        assert "input" in result

    def test_returns_full_payload_when_no_request_fields(self):
        """Should return full JSON payload when request_fields not set."""
        data = {"query": "test", "input": "value"}
        settings = PassThroughGuardrailSettings(response_fields=["output"])
        result = PassthroughGuardrailHandler.prepare_input(data, settings)
        assert "query" in result
        assert "input" in result

    def test_returns_targeted_fields(self):
        """Should return only targeted fields when request_fields set."""
        data = {"query": "targeted", "input": "also targeted", "other": "ignored"}
        settings = PassThroughGuardrailSettings(request_fields=["query", "input"])
        result = PassthroughGuardrailHandler.prepare_input(data, settings)
        assert "targeted" in result
        assert "also targeted" in result
        assert "ignored" not in result


class TestPassthroughGuardrailHandlerPrepareOutput:
    """Tests for PassthroughGuardrailHandler.prepare_output method."""

    def test_returns_full_payload_when_no_settings(self):
        """Should return full JSON payload when no settings provided."""
        data = {"results": [{"text": "result1"}], "output": "value"}
        result = PassthroughGuardrailHandler.prepare_output(data, None)
        assert "results" in result
        assert "output" in result

    def test_returns_full_payload_when_no_response_fields(self):
        """Should return full JSON payload when response_fields not set."""
        data = {"results": [{"text": "result1"}], "output": "value"}
        settings = PassThroughGuardrailSettings(request_fields=["input"])
        result = PassthroughGuardrailHandler.prepare_output(data, settings)
        assert "results" in result
        assert "output" in result

    def test_returns_targeted_fields(self):
        """Should return only targeted fields when response_fields set."""
        data = {
            "results": [{"text": "targeted1"}, {"text": "targeted2"}],
            "other": "ignored",
        }
        settings = PassThroughGuardrailSettings(response_fields=["results[*].text"])
        result = PassthroughGuardrailHandler.prepare_output(data, settings)
        assert "targeted1" in result
        assert "targeted2" in result
        assert "ignored" not in result

