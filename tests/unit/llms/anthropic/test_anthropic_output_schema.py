"""Tests for the shared Claude structured-output schema shaper."""

import copy

import pytest

from litellm.llms.anthropic.output_schema import (
    SUPPORTED_STRING_FORMATS,
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


@pytest.mark.parametrize("fmt", sorted(SUPPORTED_STRING_FORMATS))
def test_supported_string_format_is_kept(fmt):
    assert shape_claude_output_schema({"type": "string", "format": fmt}) == {"type": "string", "format": fmt}


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
