import os
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.common_utils import (
    _get_vertex_url,
    convert_anyof_null_to_nullable,
    get_vertex_location_from_url,
    get_vertex_project_id_from_url,
    set_schema_property_ordering,
)


@pytest.mark.asyncio
async def test_get_vertex_project_id_from_url():
    """Test _get_vertex_project_id_from_url with various URLs"""
    # Test with valid URL
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent"
    project_id = get_vertex_project_id_from_url(url)
    assert project_id == "test-project"

    # Test with invalid URL
    url = "https://invalid-url.com"
    project_id = get_vertex_project_id_from_url(url)
    assert project_id is None


@pytest.mark.asyncio
async def test_get_vertex_location_from_url():
    """Test _get_vertex_location_from_url with various URLs"""
    # Test with valid URL
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent"
    location = get_vertex_location_from_url(url)
    assert location == "us-central1"

    # Test with invalid URL
    url = "https://invalid-url.com"
    location = get_vertex_location_from_url(url)
    assert location is None


def test_basic_anyof_conversion():
    """Test basic conversion of anyOf with 'null'."""
    schema = {
        "type": "object",
        "properties": {"example": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
    }

    convert_anyof_null_to_nullable(schema)

    expected = {
        "type": "object",
        "properties": {"example": {"anyOf": [{"type": "string", "nullable": True}]}},
    }
    assert schema == expected


def test_nested_anyof_conversion():
    """Test nested conversion with 'anyOf' inside properties."""
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    }
                },
            }
        },
    }

    convert_anyof_null_to_nullable(schema)

    expected = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "nullable": True,
                            },
                            {"type": "string", "nullable": True},
                        ]
                    }
                },
            }
        },
    }
    assert schema == expected


def test_anyof_with_excessive_nesting():
    """Test conversion with excessive nesting > max levels +1 deep."""
    # generate a schema with excessive nesting
    schema = {"type": "object", "properties": {}}
    current = schema
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH + 1):
        current["properties"] = {
            "nested": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "properties": {},
            }
        }
        current = current["properties"]["nested"]

    # running the conversion will raise an error
    with pytest.raises(
        ValueError,
        match=f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting.",
    ):
        convert_anyof_null_to_nullable(schema)


@pytest.mark.asyncio
async def test_get_supports_system_message():
    """Test get_supports_system_message with different models"""
    from litellm.llms.vertex_ai.common_utils import get_supports_system_message

    # fine-tuned vertex gemini models will specifiy they are in the /gemini spec format
    result = get_supports_system_message(
        model="gemini/1234567890", custom_llm_provider="vertex_ai"
    )
    assert result == True

    # non-fine-tuned vertex gemini models will not specifiy they are in the /gemini spec format
    result = get_supports_system_message(
        model="random-model-name", custom_llm_provider="vertex_ai"
    )
    assert result == False


def test_set_schema_property_ordering_with_excessive_nesting():
    """Test set_schema_property_ordering with excessive nesting > max levels +1 deep."""
    # generate a schema with excessive nesting
    schema = {"type": "object", "properties": {}}
    current = schema
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH + 1):
        current["properties"] = {"nested": {"type": "object", "properties": {}}}
        current = current["properties"]["nested"]

    # running the function will raise an error
    with pytest.raises(
        ValueError,
        match=f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting.",
    ):
        set_schema_property_ordering(schema)


def test_build_vertex_schema():
    """Test build_vertex_schema with a sample schema"""
    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

    parameters = {
        "properties": {
            "state": {
                "properties": {
                    "messages": {"items": {}, "type": "array"},
                    "conversation_id": {"type": "string"},
                },
                "required": ["messages", "conversation_id"],
                "type": "object",
            },
            "config": {
                "description": "Configuration for a Runnable.",
                "properties": {
                    "tags": {"items": {"type": "string"}, "type": "array"},
                    "metadata": {"type": "object"},
                    "callbacks": {
                        "anyOf": [{"items": {}, "type": "array"}, {}, {"type": "null"}]
                    },
                    "run_name": {"type": "string"},
                    "max_concurrency": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}]
                    },
                    "recursion_limit": {"type": "integer"},
                    "configurable": {"type": "object"},
                    "run_id": {
                        "anyOf": [
                            {"format": "uuid", "type": "string"},
                            {"type": "null"},
                        ]
                    },
                },
                "type": "object",
            },
            "kwargs": {"default": None, "type": "object"},
        },
        "required": ["state", "config"],
        "type": "object",
    }

    expected_output = {
        "properties": {
            "state": {
                "properties": {
                    "messages": {"items": {"type": "object"}, "type": "array"},
                    "conversation_id": {"type": "string"},
                },
                "required": ["messages", "conversation_id"],
                "type": "object",
            },
            "config": {
                "description": "Configuration for a Runnable.",
                "properties": {
                    "tags": {"items": {"type": "string"}, "type": "array"},
                    "metadata": {"type": "object"},
                    "callbacks": {
                        "anyOf": [
                            {"type": "array", "nullable": True},
                            {"type": "object", "nullable": True},
                        ]
                    },
                    "run_name": {"type": "string"},
                    "max_concurrency": {
                        "anyOf": [{"type": "integer", "nullable": True}]
                    },
                    "recursion_limit": {"type": "integer"},
                    "configurable": {"type": "object"},
                    "run_id": {
                        "anyOf": [
                            {"format": "uuid", "type": "string", "nullable": True}
                        ]
                    },
                },
                "type": "object",
            },
            "kwargs": {"default": None, "type": "object"},
        },
        "required": ["state", "config"],
        "type": "object",
    }

    assert _build_vertex_schema(parameters) == expected_output


def test_process_items_with_excessive_nesting():
    """Test process_items with excessive nesting > max levels +1 deep."""
    # generate a schema with excessive nesting
    from litellm.llms.vertex_ai.common_utils import process_items

    schema = {"type": "object", "properties": {}}
    current = schema
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH + 1):
        current["properties"] = {"nested": {"type": "object", "properties": {}}}
        current = current["properties"]["nested"]

    with pytest.raises(
        ValueError,
        match=f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting.",
    ):
        process_items(schema)


def test_process_items_basic():
    """Test basic functionality of process_items."""
    from litellm.llms.vertex_ai.common_utils import process_items

    # Test empty items
    schema = {"type": "array", "items": {}}
    process_items(schema)
    assert schema["items"] == {"type": "object"}

    # Test nested items
    schema = {"type": "array", "items": {"type": "array", "items": {}}}
    process_items(schema)
    assert schema["items"]["items"] == {"type": "object"}

    # Test items in properties
    schema = {
        "type": "object",
        "properties": {"nested": {"type": "array", "items": {}}},
    }
    process_items(schema)
    assert schema["properties"]["nested"]["items"] == {"type": "object"}


def test_vertex_ai_complex_response_schema():
    import json
    from copy import deepcopy

    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()
    non_default_params = {
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "generic_schema",
                "schema": {
                    "$id": "https://example.com/schema",
                    "type": "object",
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "required": ["field1", "field2", "field3"],
                    "properties": {
                        "field3": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["subfield1", "subfield2"],
                                "properties": {
                                    "subfield1": {
                                        "type": "string",
                                        "description": "string field",
                                    },
                                    "subfield2": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "oneOf": [
                                                {
                                                    "title": "Type1",
                                                    "required": ["Type1"],
                                                },
                                                {
                                                    "title": "Type2",
                                                    "required": ["Type2"],
                                                },
                                                {
                                                    "title": "Type3",
                                                    "required": ["Type3"],
                                                },
                                            ],
                                            "properties": {
                                                "Type1": {
                                                    "type": "object",
                                                    "required": ["prop1", "prop2"],
                                                    "properties": {
                                                        "prop1": {
                                                            "type": "string",
                                                            "description": "string value",
                                                        },
                                                        "prop2": {
                                                            "type": "string",
                                                            "description": "string value",
                                                        },
                                                    },
                                                    "additionalProperties": False,
                                                },
                                                "Type3": {
                                                    "type": "object",
                                                    "required": ["prop1", "prop3"],
                                                    "properties": {
                                                        "prop1": {
                                                            "type": "string",
                                                            "description": "string value",
                                                        },
                                                        "prop3": {
                                                            "type": "array",
                                                            "items": {
                                                                "type": "object",
                                                                "required": [
                                                                    "item1",
                                                                    "item2",
                                                                ],
                                                                "properties": {
                                                                    "item1": {
                                                                        "type": "string",
                                                                        "description": "string value",
                                                                    },
                                                                    "item2": {
                                                                        "type": "boolean",
                                                                        "description": "boolean value",
                                                                    },
                                                                },
                                                                "additionalProperties": False,
                                                            },
                                                        },
                                                    },
                                                    "additionalProperties": False,
                                                },
                                                "Type2": {
                                                    "type": "object",
                                                    "required": ["prop1", "prop3"],
                                                    "properties": {
                                                        "prop1": {
                                                            "type": "string",
                                                            "description": "string value",
                                                        },
                                                        "prop3": {
                                                            "type": "array",
                                                            "items": {
                                                                "type": "object",
                                                                "required": [
                                                                    "item1",
                                                                    "item2",
                                                                ],
                                                                "properties": {
                                                                    "item1": {
                                                                        "type": "string",
                                                                        "description": "string value",
                                                                    },
                                                                    "item2": {
                                                                        "type": "boolean",
                                                                        "description": "boolean value",
                                                                    },
                                                                },
                                                                "additionalProperties": False,
                                                            },
                                                        },
                                                    },
                                                    "additionalProperties": False,
                                                },
                                            },
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "field1": {"type": "string", "description": "string field"},
                        "field2": {"type": "string", "description": "string field"},
                    },
                    "additionalProperties": False,
                },
                "strict": True,
                "description": "Generic schema for testing",
            },
        },
    }
    original_non_default_params = deepcopy(non_default_params)
    optional_params = {}

    v.apply_response_schema_transformation(
        value=non_default_params["response_format"], optional_params=optional_params
    )

    # Assertions for the transformed schema
    transformed_schema = optional_params["response_schema"]

    # Check top level structure
    assert transformed_schema["type"] == "object"
    assert "propertyOrdering" in transformed_schema
    assert transformed_schema["propertyOrdering"] == ["field3", "field1", "field2"]
    assert set(transformed_schema["required"]) == {"field1", "field2", "field3"}

    # Check field3 structure (array of objects)
    field3 = transformed_schema["properties"]["field3"]
    assert field3["type"] == "array"

    # Check field3 items structure
    field3_items = field3["items"]
    assert field3_items["type"] == "object"
    assert "propertyOrdering" in field3_items
    assert field3_items["propertyOrdering"] == ["subfield1", "subfield2"]
    assert set(field3_items["required"]) == {"subfield1", "subfield2"}

    # Check subfield2 structure (array of objects)
    subfield2 = field3_items["properties"]["subfield2"]
    assert subfield2["type"] == "array"

    # Check subfield2 items structure
    subfield2_items = subfield2["items"]
    assert subfield2_items["type"] == "object"
    assert "propertyOrdering" in subfield2_items
    assert subfield2_items["propertyOrdering"] == ["Type1", "Type3", "Type2"]

    # Check Type1 structure
    type1 = subfield2_items["properties"]["Type1"]
    assert type1["type"] == "object"
    assert "propertyOrdering" in type1
    assert type1["propertyOrdering"] == ["prop1", "prop2"]
    assert set(type1["required"]) == {"prop1", "prop2"}

    # Check Type2 structure
    type2 = subfield2_items["properties"]["Type2"]
    assert type2["type"] == "object"
    assert "propertyOrdering" in type2
    assert type2["propertyOrdering"] == ["prop1", "prop3"]
    assert set(type2["required"]) == {"prop1", "prop3"}

    # Check Type3 structure
    type3 = subfield2_items["properties"]["Type3"]
    assert type3["type"] == "object"
    assert "propertyOrdering" in type3
    assert type3["propertyOrdering"] == ["prop1", "prop3"]
    assert set(type3["required"]) == {"prop1", "prop3"}

    # Check nested items in Type3's prop3
    type3_prop3 = type3["properties"]["prop3"]
    assert type3_prop3["type"] == "array"
    type3_prop3_items = type3_prop3["items"]
    assert type3_prop3_items["type"] == "object"
    assert "propertyOrdering" in type3_prop3_items
    assert type3_prop3_items["propertyOrdering"] == ["item1", "item2"]
    assert set(type3_prop3_items["required"]) == {"item1", "item2"}

    # Verify additionalProperties were removed during transformation
    assert "additionalProperties" not in transformed_schema
    assert "additionalProperties" not in field3_items
    assert "additionalProperties" not in subfield2_items
    assert "additionalProperties" not in type1
    assert "additionalProperties" not in type2
    assert "additionalProperties" not in type3
    assert "additionalProperties" not in type3_prop3_items


@pytest.mark.parametrize(
    "stream, expected_endpoint_suffix",
    [
        (True, "streamGenerateContent?alt=sse"),
        (False, "generateContent"),
    ],
)
def test_get_vertex_url_global_region(stream, expected_endpoint_suffix):
    """
    Test _get_vertex_url when vertex_location is 'global' for chat mode.
    """
    mode = "chat"
    model = "gemini-1.5-pro-preview-0409"
    vertex_project = "test-g-project"
    vertex_location = "global"
    vertex_api_version = "v1"

    # Mock litellm.VertexGeminiConfig.get_model_for_vertex_ai_url to return model as is
    # as we are not testing that part here, just the URL construction
    with patch(
        "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
        side_effect=lambda model: model,
    ):
        url, endpoint = _get_vertex_url(
            mode=mode,
            model=model,
            stream=stream,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_api_version=vertex_api_version,
        )

    expected_url_base = f"https://aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/global/publishers/google/models/{model}"

    if stream:
        expected_endpoint = "streamGenerateContent"
        expected_url = f"{expected_url_base}:{expected_endpoint}?alt=sse"
    else:
        expected_endpoint = "generateContent"
        expected_url = f"{expected_url_base}:{expected_endpoint}"

    assert endpoint == expected_endpoint
    assert url == expected_url
