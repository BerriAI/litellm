"""Tests for FireworksAIConfig tool-schema sanitization.

Covers the regression where Fireworks AI rejects tool parameter schemas
containing ``title`` or ``default: None`` (both auto-emitted by Pydantic) with
a 400 "JSON Schema not supported" error. See issues #27821 and #28149.
"""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig

# --- _sanitize_tool_schema unit cases --------------------------------------


def test_sanitize_strips_top_level_title():
    schema = {"title": "PageRequest", "type": "object"}
    assert FireworksAIConfig._sanitize_tool_schema(schema) == {"type": "object"}


def test_sanitize_strips_default_none_only_keeps_other_defaults():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "default": None},
            "b": {"type": "integer", "default": 10},
            "c": {"type": "string", "default": ""},
        },
    }
    out = FireworksAIConfig._sanitize_tool_schema(schema)
    assert "default" not in out["properties"]["a"]
    assert out["properties"]["b"]["default"] == 10
    # empty string is a legitimate non-null default and must survive
    assert out["properties"]["c"]["default"] == ""


def test_sanitize_recurses_into_nested_properties():
    schema = {
        "type": "object",
        "properties": {
            "page": {
                "type": "object",
                "title": "Page",
                "properties": {
                    "page_size": {
                        "default": None,
                        "title": "Page Size",
                        "type": "integer",
                    },
                },
            }
        },
    }
    out = FireworksAIConfig._sanitize_tool_schema(schema)
    assert "title" not in out["properties"]["page"]
    assert "title" not in out["properties"]["page"]["properties"]["page_size"]
    assert "default" not in out["properties"]["page"]["properties"]["page_size"]
    assert out["properties"]["page"]["properties"]["page_size"]["type"] == "integer"


def test_sanitize_recurses_into_array_items():
    schema = {
        "type": "array",
        "items": {"type": "object", "title": "Item", "properties": {}},
    }
    out = FireworksAIConfig._sanitize_tool_schema(schema)
    assert "title" not in out["items"]


@pytest.mark.parametrize("keyword", ["anyOf", "allOf", "oneOf"])
def test_sanitize_recurses_into_composition_keywords(keyword):
    schema = {
        keyword: [
            {"type": "string", "title": "StringVariant"},
            {"type": "integer", "default": None},
        ]
    }
    out = FireworksAIConfig._sanitize_tool_schema(schema)
    assert "title" not in out[keyword][0]
    assert "default" not in out[keyword][1]
    assert out[keyword][0]["type"] == "string"
    assert out[keyword][1]["type"] == "integer"


def test_sanitize_passthrough_for_non_container_values():
    # Strings, ints, None should pass through unchanged
    assert FireworksAIConfig._sanitize_tool_schema("hello") == "hello"
    assert FireworksAIConfig._sanitize_tool_schema(42) == 42
    assert FireworksAIConfig._sanitize_tool_schema(None) is None


def test_sanitize_does_not_strip_keyed_title_inside_enum_values():
    # 'title' should be stripped only as a key; if it appears as a string
    # value (e.g. an enum entry), it must survive.
    schema = {"type": "string", "enum": ["title", "subtitle", "body"]}
    out = FireworksAIConfig._sanitize_tool_schema(schema)
    assert out["enum"] == ["title", "subtitle", "body"]


# --- _transform_tools integration cases ------------------------------------


def test_transform_tools_sanitizes_parameters_and_keeps_strict_strip():
    """End-to-end: Pydantic-shaped tool schema goes in, sanitized version
    comes out, ``strict`` is still removed at the top level."""
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_pages",
                "description": "List paginated results",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "title": "ListPagesParams",
                    "properties": {
                        "page_size": {
                            "default": None,
                            "title": "Page Size",
                            "type": "integer",
                        },
                        "cursor": {
                            "default": None,
                            "title": "Cursor",
                            "type": "string",
                        },
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "title": "Filter",
                                "properties": {
                                    "field": {"type": "string", "title": "Field"},
                                },
                            },
                        },
                    },
                    "required": [],
                },
            },
        }
    ]

    out = config._transform_tools(tools=copy.deepcopy(tools))

    fn = out[0]["function"]
    assert "strict" not in fn
    params = fn["parameters"]
    assert "title" not in params
    # Top-level + nested + array-items all sanitized
    assert "title" not in params["properties"]["page_size"]
    assert "default" not in params["properties"]["page_size"]
    assert "title" not in params["properties"]["cursor"]
    assert "title" not in params["properties"]["filters"]["items"]
    assert (
        "title" not in params["properties"]["filters"]["items"]["properties"]["field"]
    )
    # Other fields preserved
    assert params["properties"]["page_size"]["type"] == "integer"
    assert params["properties"]["filters"]["type"] == "array"


def test_transform_tools_noop_on_non_function_tools():
    """Tools whose type is not 'function' should pass through unchanged."""
    config = FireworksAIConfig()
    tools = [{"type": "code_interpreter"}]
    assert config._transform_tools(tools=copy.deepcopy(tools)) == tools


def test_transform_tools_handles_function_with_no_parameters():
    """Some MCP tools have no ``parameters`` field — must not KeyError."""
    config = FireworksAIConfig()
    tools = [{"type": "function", "function": {"name": "ping", "strict": True}}]
    out = config._transform_tools(tools=copy.deepcopy(tools))
    assert "strict" not in out[0]["function"]
    assert "parameters" not in out[0]["function"]
