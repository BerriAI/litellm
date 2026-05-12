"""
Unit tests for the OpenTelemetry dict/list JSON encoding fix.

Verifies that _cast_as_primitive_value_type serialises dict and list values
using valid JSON (safe_dumps), not Python repr (str()).

Regression: before the fix, {"key": "value"} was emitted as {'key': 'value'}
in OTEL span attributes, making spend_logs_metadata unparseable in downstream
observability tools (Jaeger, Honeycomb, Grafana Tempo).

Fixes: https://github.com/BerriAI/litellm/issues/27451
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))


def _get_otel_logger():
    """Return an OpenTelemetryLogger instance without any real OTEL dependency."""
    from litellm.integrations.opentelemetry import OpenTelemetryLogger

    return OpenTelemetryLogger.__new__(OpenTelemetryLogger)


class TestCastAsPrimitiveValueType:
    """_cast_as_primitive_value_type must produce valid JSON for dict/list."""

    def test_dict_produces_valid_json_not_repr(self):
        """
        A dict value must be JSON-encoded, not Python-repr'd.
        Python repr uses single quotes; JSON uses double quotes.
        """
        logger = _get_otel_logger()
        result = logger._cast_as_primitive_value_type({"key": "value"})

        # Must be parseable as JSON
        parsed = json.loads(result)
        assert parsed == {"key": "value"}, f"Unexpected parsed value: {parsed}"

        # Must NOT be Python repr (single-quoted keys)
        assert "'key'" not in result, (
            f"Result looks like Python repr, not JSON: {result!r}"
        )

    def test_nested_dict_produces_valid_json(self):
        """Nested dicts (e.g. spend_logs_metadata with custom keys) must JSON-encode."""
        logger = _get_otel_logger()
        payload = {"env": "prod", "team": "platform", "request_id": "abc-123"}
        result = logger._cast_as_primitive_value_type(payload)

        parsed = json.loads(result)
        assert parsed == payload

    def test_list_produces_valid_json_not_repr(self):
        """List values must also be JSON-encoded."""
        logger = _get_otel_logger()
        result = logger._cast_as_primitive_value_type(["a", "b", "c"])

        parsed = json.loads(result)
        assert parsed == ["a", "b", "c"]

    def test_primitives_unchanged(self):
        """str, bool, int, float must pass through without modification."""
        logger = _get_otel_logger()
        assert logger._cast_as_primitive_value_type("hello") == "hello"
        assert logger._cast_as_primitive_value_type(True) is True
        assert logger._cast_as_primitive_value_type(42) == 42
        assert logger._cast_as_primitive_value_type(3.14) == 3.14

    def test_none_returns_empty_string(self):
        """None must return empty string (existing behaviour must be preserved)."""
        logger = _get_otel_logger()
        assert logger._cast_as_primitive_value_type(None) == ""

    def test_dict_with_nested_list_produces_valid_json(self):
        """Real-world spend_logs_metadata may contain lists inside dicts."""
        logger = _get_otel_logger()
        payload = {"tags": ["prod", "us-east"], "version": 2}
        result = logger._cast_as_primitive_value_type(payload)

        parsed = json.loads(result)
        assert parsed == payload
