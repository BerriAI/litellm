"""
Tests for null byte sanitization in spend log data before PostgreSQL insertion.

PostgreSQL text columns reject null bytes (\u0000) with error 22P05:
"unsupported Unicode escape sequence". This module validates that
_strip_null_bytes() and jsonify_object() properly sanitize data.

Refs: https://github.com/BerriAI/litellm/issues/15519
"""

import json

import pytest

from litellm.proxy.utils import _strip_null_bytes, jsonify_object


class TestStripNullBytes:
    """Unit tests for _strip_null_bytes()."""

    def test_strips_null_from_plain_string(self):
        assert _strip_null_bytes("hello\x00world") == "helloworld"

    def test_strips_multiple_null_bytes(self):
        assert _strip_null_bytes("\x00a\x00b\x00") == "ab"

    def test_no_op_on_clean_string(self):
        assert _strip_null_bytes("hello world") == "hello world"

    def test_handles_empty_string(self):
        assert _strip_null_bytes("") == ""

    def test_strips_from_dict_values(self):
        data = {"key": "val\x00ue", "other": "clean"}
        result = _strip_null_bytes(data)
        assert result == {"key": "value", "other": "clean"}

    def test_strips_from_nested_dict(self):
        data = {"outer": {"inner": "has\x00null"}}
        result = _strip_null_bytes(data)
        assert result == {"outer": {"inner": "hasnull"}}

    def test_strips_from_list(self):
        data = ["a\x00b", "c\x00d"]
        result = _strip_null_bytes(data)
        assert result == ["ab", "cd"]

    def test_strips_from_list_in_dict(self):
        data = {"messages": ["hello\x00", "world\x00"]}
        result = _strip_null_bytes(data)
        assert result == {"messages": ["hello", "world"]}

    def test_preserves_non_string_types(self):
        assert _strip_null_bytes(42) == 42
        assert _strip_null_bytes(3.14) == 3.14
        assert _strip_null_bytes(True) is True
        assert _strip_null_bytes(None) is None


class TestJsonifyObjectNullBytes:
    """Integration tests for jsonify_object() null byte handling."""

    def test_strips_null_from_top_level_string_field(self):
        data = {"request_id": "req\x00123", "model": "gpt-4o"}
        result = jsonify_object(data)
        assert result["request_id"] == "req123"

    def test_strips_null_from_serialized_dict_field(self):
        data = {
            "metadata": {"user_msg": "prompt with \x00 null byte"},
            "model": "gpt-4o",
        }
        result = jsonify_object(data)
        parsed = json.loads(result["metadata"])
        assert "\x00" not in parsed["user_msg"]

    def test_strips_null_from_response_content(self):
        """Simulate a spend log entry with null bytes in the LLM response."""
        data = {
            "request_id": "msg_01BT4f5Rkt7ub18gYkexXEv1",
            "call_type": "acompletion",
            "model": "claude-sonnet-4-20250514",
            "spend": 0.13794,
            "messages": {"content": "response with \x00 embedded"},
            "response": {"text": "output\x00here"},
        }
        result = jsonify_object(data)
        assert "\x00" not in result["messages"]
        assert "\x00" not in result["response"]

    def test_original_data_not_mutated(self):
        """jsonify_object uses deepcopy — original must stay intact."""
        data = {"key": "has\x00null"}
        jsonify_object(data)
        assert data["key"] == "has\x00null"
