"""Tests for the shared Claude structured-output schema shaper."""

import copy

import pytest

from litellm.llms.anthropic.output_schema import (
    shape_claude_output_format,
    shape_claude_output_schema,
)

_MESSY_SCHEMA = {
    "type": "object",
    "description": "A record",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3, "uniqueItems": True},
        "score": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
        "id": {"type": "string", "format": "int64", "minLength": 1},
        "meta": {"type": ["object", "null"], "properties": {"a": {"type": "string"}}},
        "kind": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        "labels": {"type": "object", "additionalProperties": {"type": "string"}},
        "nested": {"properties": {"b": {"type": "boolean"}}},
    },
    "required": ["tags", "score", "id"],
    "additionalProperties": True,
}


def test_shaping_is_idempotent():
    once = shape_claude_output_schema(_MESSY_SCHEMA)

    assert shape_claude_output_schema(once) == once


def test_shaping_does_not_mutate_the_input():
    before = copy.deepcopy(_MESSY_SCHEMA)

    shape_claude_output_schema(_MESSY_SCHEMA)

    assert _MESSY_SCHEMA == before


# Claude-supported formats per the anthropic SDK ``lib/_parse/_transform.py`` SupportedStringFormats, checked 2026-10-13
@pytest.mark.parametrize("fmt", ["date-time", "date", "email", "uri", "uuid"])
def test_supported_string_format_is_kept(fmt: str):
    assert shape_claude_output_schema({"type": "string", "format": fmt}) == {"type": "string", "format": fmt}


_BOUNDED = {"type": "number", "minimum": 0}
_BOUNDED_SHAPED = {"type": "number", "description": "Note: minimum value: 0."}


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"$defs": {"X": _BOUNDED}}, {"$defs": {"X": _BOUNDED_SHAPED}}),
        ({"definitions": {"X": _BOUNDED}}, {"definitions": {"X": _BOUNDED_SHAPED}}),
        ({"allOf": [_BOUNDED]}, {"allOf": [_BOUNDED_SHAPED]}),
    ],
    ids=["$defs", "definitions", "allOf"],
)
def test_shapes_schemas_nested_under_definitions_and_all_of(schema: dict[str, object], expected: dict[str, object]):
    assert shape_claude_output_schema(schema) == expected


@pytest.mark.parametrize("composition", ["anyOf", "oneOf", "allOf", "$ref"])
def test_untyped_node_with_properties_and_composition_is_not_forced_to_object(composition: str):
    value = "#/$defs/X" if composition == "$ref" else [{"type": "object", "properties": {}}]
    result = shape_claude_output_schema({"properties": {"a": {"type": "string"}}, composition: value})

    assert "type" not in result
    assert "additionalProperties" not in result


def test_output_format_block_shapes_only_its_schema():
    block = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}},
    }

    result = shape_claude_output_format(block)

    assert result == {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "Note: minimum value: 1."}},
            "additionalProperties": False,
        },
    }
    assert block["schema"]["properties"]["n"] == {"type": "integer", "minimum": 1}


def test_output_format_block_without_schema_is_returned_unchanged():
    assert shape_claude_output_format({"type": "json_schema"}) == {"type": "json_schema"}
