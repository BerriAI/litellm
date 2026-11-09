import json

import pytest

from litellm.litellm_core_utils.prompt_templates.tool_call_json_repair import (
    parse_tool_call_arguments,
    salvage_concatenated_tool_arguments,
    split_concatenated_json_objects,
)


def test_split_concatenated_json_single_object():
    """A single valid JSON object is returned as a one-element list."""
    result = split_concatenated_json_objects('{"location": "Boston"}')
    assert result == [{"location": "Boston"}]


def test_split_concatenated_json_multiple_objects():
    """
    Multiple JSON objects concatenated without separators are split correctly.
    This is the exact pattern from issue #20543 where Bedrock Claude Sonnet 4.5
    returns concatenated JSON in a single tool call arguments string.
    """
    raw = (
        '{"command": ["curl", "-i", "http://localhost:9009"]}'
        '{"command": ["curl", "-i", "http://localhost:9009/robots.txt"]}'
        '{"command": ["curl", "-i", "http://localhost:9009/sitemap.xml"]}'
    )
    result = split_concatenated_json_objects(raw)
    assert len(result) == 3
    assert result[0] == {"command": ["curl", "-i", "http://localhost:9009"]}
    assert result[1] == {"command": ["curl", "-i", "http://localhost:9009/robots.txt"]}
    assert result[2] == {"command": ["curl", "-i", "http://localhost:9009/sitemap.xml"]}


def test_split_concatenated_json_with_whitespace():
    """Objects separated by whitespace are handled correctly."""
    raw = '{"a": 1}  {"b": 2}\n{"c": 3}'
    result = split_concatenated_json_objects(raw)
    assert len(result) == 3
    assert result[0] == {"a": 1}
    assert result[1] == {"b": 2}
    assert result[2] == {"c": 3}


def test_split_concatenated_json_empty_string():
    """Empty or whitespace-only strings return an empty list."""
    assert split_concatenated_json_objects("") == []
    assert split_concatenated_json_objects("   ") == []


def test_split_concatenated_json_non_dict_value():
    """Non-dict JSON values (e.g. arrays, strings) are replaced with {}."""
    result = split_concatenated_json_objects("[1, 2, 3]")
    assert result == [{}]


def test_split_concatenated_json_wholly_invalid_returns_empty():
    """
    Wholly unparseable JSON degrades to an empty list instead of raising.

    Regression for https://github.com/BerriAI/litellm/issues/18667: a raise
    here propagated out of `_convert_to_bedrock_tool_call_invoke` and turned
    every replayed conversation into a 500.
    """
    assert split_concatenated_json_objects("not json at all") == []


def test_split_concatenated_json_malformed_object_returns_empty():
    """
    A single malformed object (missing comma between keys) degrades to an
    empty list rather than raising `Expecting ',' delimiter`.
    """
    assert split_concatenated_json_objects('{"location": "Boston" "unit": "celsius"}') == []


def test_split_concatenated_json_salvages_prefix_before_truncated_tail():
    """
    Complete objects parsed before an unparseable/truncated tail are kept;
    only the bad tail is discarded.
    """
    result = split_concatenated_json_objects('{"a": 1}{"b": 2}{"c":')
    assert result == [{"a": 1}, {"b": 2}]


def test_parse_tool_call_arguments_rejects_concatenated_json() -> None:
    with pytest.raises(ValueError, match="Failed to parse tool call arguments"):
        parse_tool_call_arguments('{"a":1}{"b":2}')


def _distinct_json_objects(count: int) -> str:
    return "".join(json.dumps({"n": index}, separators=(",", ":")) for index in range(count))


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ('{"a":1}{"b":2}', ({"a": 1}, {"b": 2})),
        ('{"a":1}{"a":1}{"a":1}', ({"a": 1},)),
        ('{"a":1}{"a":1}{"b":2}', ({"a": 1}, {"a": 1}, {"b": 2})),
        (_distinct_json_objects(8), tuple({"n": index} for index in range(8))),
        (_distinct_json_objects(9), ()),
        (_distinct_json_objects(9) + " junk", ()),
        ('{"a":1}' * 7 + '{"b":2}', tuple({"a": 1} for _ in range(7)) + ({"b": 2},)),
        ('{"a":1}' * 8 + '{"b":2}', ()),
        ('{"a":1}' * 5000, ({"a": 1},)),
        ('{"a":1}' * 20, ({"a": 1},)),
        ('{"a":1}{"b":', ()),
        ('0{"x":1}', ()),
        ('{"x":1}0', ()),
        ('[1]{"x":1}', ()),
        ('{"a":1}{"b":2}}', ()),
        ('{"a":1} junk', ()),
    ),
)
def test_salvage_concatenated_tool_arguments(raw: str, expected: tuple[dict[str, object], ...]) -> None:
    assert salvage_concatenated_tool_arguments(raw) == expected
