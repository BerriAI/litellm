"""Tests for Fireworks AI tool schema sanitization.

Fireworks AI rejects JSON Schema properties containing 'default': null or
'title' fields.  The _sanitize_schema helper and _transform_tools must strip
these before the request is sent.

Relevant issue: https://github.com/BerriAI/litellm/issues/27821
"""

import copy

import pytest

from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig


@pytest.fixture
def config():
    return FireworksAIConfig()


class TestSanitizeSchema:
    """Unit tests for FireworksAIConfig._sanitize_schema."""

    def test_removes_title_from_top_level(self):
        schema = {"type": "object", "title": "MyParams"}
        result = FireworksAIConfig._sanitize_schema(schema)
        assert "title" not in result

    def test_removes_default_none(self):
        schema = {"type": "string", "default": None}
        result = FireworksAIConfig._sanitize_schema(schema)
        assert "default" not in result

    def test_keeps_default_non_none(self):
        schema = {"type": "integer", "default": 42}
        result = FireworksAIConfig._sanitize_schema(schema)
        assert result["default"] == 42

    def test_removes_title_and_default_none_from_nested_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "page_size": {
                    "type": "integer",
                    "title": "Page Size",
                    "default": None,
                },
                "query": {
                    "type": "string",
                    "title": "Query",
                    "default": "hello",
                },
            },
        }
        result = FireworksAIConfig._sanitize_schema(schema)
        assert "title" not in result["properties"]["page_size"]
        assert "default" not in result["properties"]["page_size"]
        assert "title" not in result["properties"]["query"]
        # non-None default preserved
        assert result["properties"]["query"]["default"] == "hello"

    def test_recurses_into_items(self):
        schema = {
            "type": "array",
            "items": {"type": "string", "title": "Item", "default": None},
        }
        result = FireworksAIConfig._sanitize_schema(schema)
        assert "title" not in result["items"]
        assert "default" not in result["items"]

    def test_recurses_into_anyof(self):
        schema = {
            "anyOf": [
                {"type": "string", "title": "StrOption"},
                {"type": "null", "title": "NullOption", "default": None},
            ]
        }
        result = FireworksAIConfig._sanitize_schema(schema)
        for sub in result["anyOf"]:
            assert "title" not in sub
            assert "default" not in sub

    def test_recurses_into_allof(self):
        schema = {
            "allOf": [
                {"type": "object", "title": "Base", "properties": {"x": {"title": "X"}}}
            ]
        }
        result = FireworksAIConfig._sanitize_schema(schema)
        assert "title" not in result["allOf"][0]
        assert "title" not in result["allOf"][0]["properties"]["x"]

    def test_recurses_into_oneof(self):
        schema = {
            "oneOf": [
                {"type": "string", "title": "A"},
                {"type": "integer", "title": "B"},
            ]
        }
        result = FireworksAIConfig._sanitize_schema(schema)
        for sub in result["oneOf"]:
            assert "title" not in sub

    def test_recurses_into_defs(self):
        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "title": "Address",
                    "properties": {
                        "street": {"type": "string", "title": "Street"},
                    },
                }
            },
            "type": "object",
            "properties": {
                "home": {"$ref": "#/$defs/Address", "title": "Home"},
            },
        }
        result = FireworksAIConfig._sanitize_schema(schema)
        assert "title" not in result["$defs"]["Address"]
        assert "title" not in result["$defs"]["Address"]["properties"]["street"]
        assert "title" not in result["properties"]["home"]

    def test_deeply_nested(self):
        schema = {
            "type": "object",
            "title": "Root",
            "properties": {
                "items": {
                    "type": "array",
                    "title": "Items",
                    "items": {
                        "type": "object",
                        "title": "Item",
                        "properties": {
                            "name": {
                                "type": "string",
                                "title": "Name",
                                "default": None,
                            },
                            "tags": {
                                "anyOf": [
                                    {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "title": "Tag",
                                        },
                                    },
                                    {"type": "null"},
                                ],
                                "title": "Tags",
                                "default": None,
                            },
                        },
                    },
                }
            },
        }
        result = FireworksAIConfig._sanitize_schema(schema)
        # Check all titles and null defaults are gone
        assert "title" not in result
        items_prop = result["properties"]["items"]
        assert "title" not in items_prop
        item_schema = items_prop["items"]
        assert "title" not in item_schema
        name_prop = item_schema["properties"]["name"]
        assert "title" not in name_prop
        assert "default" not in name_prop
        tags_prop = item_schema["properties"]["tags"]
        assert "title" not in tags_prop
        assert "default" not in tags_prop
        assert "title" not in tags_prop["anyOf"][0]["items"]

    def test_preserves_other_fields(self):
        schema = {
            "type": "object",
            "description": "A test schema",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name",
                    "minLength": 1,
                    "title": "Name",
                }
            },
        }
        result = FireworksAIConfig._sanitize_schema(schema)
        assert result["type"] == "object"
        assert result["description"] == "A test schema"
        assert result["required"] == ["name"]
        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["name"]["description"] == "The name"
        assert result["properties"]["name"]["minLength"] == 1
        assert "title" not in result["properties"]["name"]


class TestTransformToolsSanitization:
    """Integration tests: _transform_tools applies schema sanitization."""

    def test_transform_tools_strips_schema_fields(self, config):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for items",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "title": "SearchParams",
                        "properties": {
                            "query": {"type": "string", "title": "Query"},
                            "page_size": {
                                "type": "integer",
                                "title": "Page Size",
                                "default": None,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        tools_copy = copy.deepcopy(tools)
        result = config._transform_tools(tools_copy)

        func = result[0]["function"]
        assert "strict" not in func
        params = func["parameters"]
        assert "title" not in params
        assert "title" not in params["properties"]["query"]
        assert "title" not in params["properties"]["page_size"]
        assert "default" not in params["properties"]["page_size"]
        # required and other fields preserved
        assert params["required"] == ["query"]

    def test_transform_tools_no_parameters(self, config):
        """Tools without parameters should not error."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "Does nothing",
                },
            }
        ]
        result = config._transform_tools(copy.deepcopy(tools))
        assert result[0]["function"]["name"] == "noop"

    def test_transform_tools_non_function_type(self, config):
        """Non-function tool types should pass through unchanged."""
        tools = [{"type": "code_interpreter"}]
        result = config._transform_tools(copy.deepcopy(tools))
        assert result == [{"type": "code_interpreter"}]
