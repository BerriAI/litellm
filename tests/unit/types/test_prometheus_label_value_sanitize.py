import pytest

from litellm.types.integrations.prometheus import (
    _sanitize_prometheus_label_value,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", ""),
        ("plain", "plain"),
        # Newlines -> spaces, carriage returns removed
        ("a\nb", "a b"),
        ("a\rb", "ab"),
        ("a\r\nb", "a b"),
        # Unicode line/paragraph separators removed
        ("a\u2028b", "ab"),
        ("a\u2029b", "ab"),
        ("a\u2028b\u2029c", "abc"),
        # Escapes per Prometheus text format
        ('he said "hi"', 'he said \\"hi\\"'),
        (r"path\to\file", r"path\\to\\file"),
        (r'quote\"slash\\', r'quote\\\"slash\\\\'),
        # Non-string inputs get coerced to str first
        (123, "123"),
        (True, "True"),
        (False, "False"),
    ],
)
def test_sanitize_prometheus_label_value_expected_outputs(value, expected):
    assert _sanitize_prometheus_label_value(value) == expected

