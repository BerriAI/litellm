import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig


@pytest.fixture(autouse=True)
def force_local_model_cost(monkeypatch):
    """Force local model cost map usage for all tests in this file."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

    litellm.model_cost = get_model_cost_map(url=litellm.model_cost_map_url)


# ---------------------------------------------------------------------------
# _sanitize_tool_schema unit tests
# ---------------------------------------------------------------------------


def test_sanitize_strips_pattern():
    """pattern is stripped from every string property to avoid
    "Conflict in schema definitions for key 'pattern'" 400s."""
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[^\\n\\r]*$"},
        },
    }
    config._sanitize_tool_schema(schema)
    assert "pattern" not in schema["properties"]["name"]
    assert schema["properties"]["name"]["type"] == "string"


def test_sanitize_strips_title():
    """title (auto-emitted by Pydantic) is stripped everywhere."""
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "properties": {
            "page_size": {"type": "integer", "title": "Page Size"},
        },
    }
    config._sanitize_tool_schema(schema)
    assert "title" not in schema["properties"]["page_size"]


def test_sanitize_strips_default_null():
    """default: null is stripped; non-null defaults are preserved."""
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "default": None},
            "b": {"type": "integer", "default": 10},
            "c": {"type": "boolean", "default": False},
        },
    }
    config._sanitize_tool_schema(schema)
    assert "default" not in schema["properties"]["a"]
    assert schema["properties"]["b"]["default"] == 10
    # False is not None — must survive
    assert schema["properties"]["c"]["default"] is False


def test_sanitize_recurses_into_nested_properties():
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {"type": "string", "pattern": "^[a-z]+$", "title": "Inner"},
                },
            },
        },
    }
    config._sanitize_tool_schema(schema)
    inner = schema["properties"]["outer"]["properties"]["inner"]
    assert "pattern" not in inner
    assert "title" not in inner


def test_sanitize_recurses_into_array_items():
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[A-Z]+$"},
            },
        },
    }
    config._sanitize_tool_schema(schema)
    assert "pattern" not in schema["properties"]["tags"]["items"]


def test_sanitize_recurses_into_anyof():
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "properties": {
            "val": {
                "anyOf": [
                    {"type": "string", "pattern": "^[0-9]+$"},
                    {"type": "integer"},
                ],
            },
        },
    }
    config._sanitize_tool_schema(schema)
    assert "pattern" not in schema["properties"]["val"]["anyOf"][0]


def test_sanitize_recurses_into_allof():
    config = FireworksAIConfig()
    schema = {
        "allOf": [
            {"type": "string", "pattern": "^[a-z]+$"},
            {"title": "Foo"},
        ],
    }
    config._sanitize_tool_schema(schema)
    assert "pattern" not in schema["allOf"][0]
    assert "title" not in schema["allOf"][1]


def test_sanitize_recurses_into_oneof():
    config = FireworksAIConfig()
    schema = {
        "oneOf": [
            {"type": "string", "pattern": "^[a-z]+$"},
            {"type": "string", "pattern": "^[A-Z]+$"},
        ],
    }
    config._sanitize_tool_schema(schema)
    for item in schema["oneOf"]:
        assert "pattern" not in item


def test_sanitize_recurses_into_dollar_defs():
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}},
        "$defs": {
            "A": {"type": "string", "pattern": "^[a-z]+$", "title": "A"},
        },
    }
    config._sanitize_tool_schema(schema)
    assert "pattern" not in schema["$defs"]["A"]
    assert "title" not in schema["$defs"]["A"]


def test_sanitize_preserves_other_fields():
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "properties": {
            "color": {
                "type": "string",
                "enum": ["red", "green", "blue"],
                "description": "Pick a color",
                "pattern": "^[a-z]+$",
            },
        },
        "required": ["color"],
        "additionalProperties": False,
    }
    config._sanitize_tool_schema(schema)
    prop = schema["properties"]["color"]
    assert prop["enum"] == ["red", "green", "blue"]
    assert prop["description"] == "Pick a color"
    assert "pattern" not in prop
    assert schema["required"] == ["color"]
    assert schema["additionalProperties"] is False


def test_sanitize_noop_on_non_dict():
    config = FireworksAIConfig()
    # Should not raise
    config._sanitize_tool_schema(None)
    config._sanitize_tool_schema("string")
    config._sanitize_tool_schema(42)
    config._sanitize_tool_schema([])


def test_sanitize_handles_empty_schema():
    config = FireworksAIConfig()
    schema = {}
    config._sanitize_tool_schema(schema)
    assert schema == {}


def test_sanitize_deeply_nested():
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {
                    "b": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "c": {"type": "string", "pattern": "^[a-z]+$"},
                            },
                        },
                    },
                },
            },
        },
    }
    config._sanitize_tool_schema(schema)
    deep = schema["properties"]["a"]["properties"]["b"]["items"]["properties"]["c"]
    assert "pattern" not in deep


def test_sanitize_recurses_into_propertyNames():
    """propertyNames subschema must be traversed (Greptile P1)."""
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "propertyNames": {
            "type": "string",
            "pattern": "^[a-z]+$",
            "title": "Prop Name",
        },
    }
    config._sanitize_tool_schema(schema)
    assert "pattern" not in schema["propertyNames"]
    assert "title" not in schema["propertyNames"]


def test_sanitize_recurses_into_prefixItems():
    """prefixItems array must be traversed."""
    config = FireworksAIConfig()
    schema = {
        "type": "array",
        "prefixItems": [
            {"type": "string", "pattern": "^[A-Z]+$"},
            {"type": "string", "title": "Second"},
        ],
    }
    config._sanitize_tool_schema(schema)
    assert "pattern" not in schema["prefixItems"][0]
    assert "title" not in schema["prefixItems"][1]


def test_sanitize_recurses_into_definitions():
    """Legacy draft-04 definitions must be traversed."""
    config = FireworksAIConfig()
    schema = {
        "type": "object",
        "definitions": {
            "Foo": {"type": "string", "pattern": "^[a-z]+$", "title": "Foo"},
        },
    }
    config._sanitize_tool_schema(schema)
    assert "pattern" not in schema["definitions"]["Foo"]
    assert "title" not in schema["definitions"]["Foo"]


# ---------------------------------------------------------------------------
# _transform_tools integration tests
# ---------------------------------------------------------------------------


def test_transform_tools_strips_pattern_from_params():
    """End-to-end: _transform_tools must strip pattern from tool parameters."""
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search things",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "pattern": "^[^\\n\\r]*$",
                            "title": "Query",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    out = config._transform_tools(tools)
    params = out[0]["function"]["parameters"]
    assert "pattern" not in params["properties"]["query"]
    assert "title" not in params["properties"]["query"]
    assert params["properties"]["query"]["type"] == "string"


def test_transform_tools_strips_default_null_from_params():
    """End-to-end: _transform_tools must strip default: null."""
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_items",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_size": {"type": "integer", "default": None, "title": "Page Size"},
                    },
                },
            },
        }
    ]
    out = config._transform_tools(tools)
    prop = out[0]["function"]["parameters"]["properties"]["page_size"]
    assert "default" not in prop
    assert "title" not in prop


def test_transform_tools_preserves_non_null_default():
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_config",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "retries": {"type": "integer", "default": 3},
                    },
                },
            },
        }
    ]
    out = config._transform_tools(tools)
    assert out[0]["function"]["parameters"]["properties"]["retries"]["default"] == 3


def test_transform_tools_noop_on_clean_schema():
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    out = config._transform_tools(tools)
    assert out[0]["function"]["parameters"] == tools[0]["function"]["parameters"]


def test_transform_tools_skips_non_function_tools_sanitization():
    """Non-function tools must pass through untouched."""
    config = FireworksAIConfig()
    non_function_tool = {
        "type": "code_interpreter",
        "code_interpreter": {"some": "config"},
    }
    out = config._transform_tools([non_function_tool])
    assert out[0] == non_function_tool


def test_transform_tools_handles_tools_without_parameters():
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "no_params_tool",
                "description": "A tool with no parameters field",
            },
        }
    ]
    # Should not raise
    out = config._transform_tools(tools)
    assert out[0]["function"]["name"] == "no_params_tool"


def test_transform_tools_strips_strict_and_pattern_together():
    """Both strict pop and pattern strip happen in the same pass."""
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "strict_tool",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string", "pattern": "^[a-z]+$"},
                    },
                },
            },
        }
    ]
    out = config._transform_tools(tools)
    assert "strict" not in out[0]["function"]
    assert "pattern" not in out[0]["function"]["parameters"]["properties"]["x"]
