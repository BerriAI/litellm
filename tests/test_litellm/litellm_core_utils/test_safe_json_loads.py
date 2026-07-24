"""
Unit tests for litellm.litellm_core_utils.safe_json_loads.safe_json_loads.

safe_json_loads parses a JSON string and, on any failure, returns a default
value instead of raising. These tests cover the success path, the fallback
path, and the non-string edge case.
"""

import pytest

from litellm.litellm_core_utils.safe_json_loads import safe_json_loads


def test_parses_valid_json_object():
    assert safe_json_loads('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_parses_valid_json_array():
    assert safe_json_loads("[1, 2, 3]") == [1, 2, 3]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('"hello"', "hello"),
        ("42", 42),
        ("3.14", 3.14),
        ("true", True),
        ("false", False),
    ],
)
def test_parses_valid_json_scalars(raw, expected):
    assert safe_json_loads(raw) == expected


def test_parses_json_null_to_none():
    # Use a non-None default so a correct parse (real None) is distinguishable
    # from the function silently falling back to its default.
    sentinel = object()
    assert safe_json_loads("null", default=sentinel) is None


def test_returns_default_none_on_invalid_json():
    assert safe_json_loads("{not valid json") is None


def test_returns_custom_default_on_invalid_json():
    sentinel = {"fallback": True}
    assert safe_json_loads("oops", default=sentinel) is sentinel


def test_returns_default_on_empty_string():
    assert safe_json_loads("") is None


@pytest.mark.parametrize("bad_input", [None, 123, 4.5, ["a"], {"b": 1}])
def test_returns_default_on_non_string_input(bad_input):
    assert safe_json_loads(bad_input, default="fallback") == "fallback"
