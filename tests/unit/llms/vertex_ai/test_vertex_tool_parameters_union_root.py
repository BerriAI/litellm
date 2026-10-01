from typing import Final

from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.vertex_ai.common_utils import _build_vertex_schema
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)


def test_build_vertex_schema_merges_object_union_root_for_tool_parameters():
    schema: Final = {
        "anyOf": [
            {"type": "object", "properties": {"query": {"type": "string"}}},
            {"type": "object", "properties": {"url": {"type": "string"}}},
        ]
    }

    result: Final = _build_vertex_schema(schema, enforce_object_root=True)

    assert result["type"] == "object"
    assert "anyOf" not in result
    assert set(result["properties"]) == {"query", "url"}


def test_build_vertex_schema_combines_conflicting_branch_property_schemas():
    schema: Final = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "lang": {"type": "string"},
                },
            },
            {
                "type": "object",
                "properties": {
                    "target": {"type": "integer"},
                    "lang": {"type": "string"},
                },
            },
        ]
    }

    result: Final = _build_vertex_schema(schema, enforce_object_root=True)

    assert result["properties"]["lang"] == {"type": "string"}
    assert result["properties"]["target"] == {"anyOf": [{"type": "string"}, {"type": "integer"}]}


def test_build_vertex_schema_keeps_only_required_fields_shared_by_every_branch():
    schema: Final = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "lang": {"type": "string"},
                },
                "required": ["query", "lang"],
            },
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "lang": {"type": "string"},
                },
                "required": ["url", "lang"],
            },
        ]
    }

    result: Final = _build_vertex_schema(schema, enforce_object_root=True)

    assert result["required"] == ["lang"]


def test_build_vertex_schema_leaves_non_object_union_root_alone():
    schema: Final = {
        "anyOf": [
            {"type": "string"},
            {"type": "object", "properties": {"url": {"type": "string"}}},
        ]
    }

    result: Final = _build_vertex_schema(schema, enforce_object_root=True)

    assert "anyOf" in result


def test_build_vertex_schema_leaves_response_schema_root_untouched():
    schema: Final = {"anyOf": [{"type": "object", "properties": {"query": {"type": "string"}}}]}

    result: Final = _build_vertex_schema(schema)

    assert "type" not in result


def test_build_vertex_schema_still_strips_fields_beside_nested_anyof():
    schema: Final = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": "x",
            }
        },
    }

    result: Final = _build_vertex_schema(schema, enforce_object_root=True)

    assert "default" not in result["properties"]["value"]


def test_vertex_ai_map_tool_with_top_level_anyof():
    v: Final = VertexGeminiConfig()
    value: Final = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search by query or by url",
                "parameters": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                        {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                        },
                    ]
                },
            },
        }
    ]

    tools: Final = v._map_function(value=value, optional_params={})

    parameters: Final = tools[0]["function_declarations"][0]["parameters"]
    assert parameters["type"] == "object"
    assert "anyOf" not in parameters
    assert set(parameters["properties"]) == {"query", "url"}


def test_google_ai_studio_map_tool_preserves_top_level_anyof():
    g: Final = GoogleAIStudioGeminiConfig()
    value: Final = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search by query or by url",
                "parameters": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                        {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                        },
                    ]
                },
            },
        }
    ]

    tools: Final = g._map_function(value=value, optional_params={})

    parameters: Final = tools[0]["function_declarations"][0]["parameters"]
    assert "anyOf" in parameters
    assert "type" not in parameters


def test_build_vertex_schema_skips_merge_when_branch_limit_exceeded():
    schema: Final = {"anyOf": [{"type": "object", "properties": {f"field_{i}": {"type": "string"}}} for i in range(65)]}

    result: Final = _build_vertex_schema(schema, enforce_object_root=True)

    assert "anyOf" in result
    assert "type" not in result
