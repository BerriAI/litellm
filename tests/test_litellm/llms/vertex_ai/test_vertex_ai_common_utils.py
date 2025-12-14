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
                    "run_id": {"anyOf": [{"type": "string", "nullable": True}]},
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


@pytest.mark.parametrize(
    "supported_regions, expected_result",
    [
        (None, False),  # get_supported_regions returns None
        ([], False),  # empty list, no global region
        (["us-central1"], False),  # only regional, no global
        (["global"], True),  # only global region
        (["global", "us-central1"], True),  # global and other regions
        (
            ["us-central1", "global", "europe-west1"],
            True,
        ),  # global among multiple regions
    ],
)
def test_is_global_only_vertex_model(supported_regions, expected_result):
    """Test is_global_only_vertex_model with various supported regions scenarios"""
    from litellm.llms.vertex_ai.common_utils import is_global_only_vertex_model

    with patch("litellm.utils.get_supported_regions") as mock_get_supported_regions:
        mock_get_supported_regions.return_value = supported_regions

        result = is_global_only_vertex_model("test-model")

        assert result == expected_result
        mock_get_supported_regions.assert_called_once_with(
            model="test-model", custom_llm_provider="vertex_ai"
        )


@pytest.mark.parametrize(
    "model_is_global_only, vertex_region, expected_region",
    [
        (True, None, "global"),  # Global-only model with no region specified
        (True, "us-central1", "global"),  # Global-only model overrides specified region
        (True, "europe-west1", "global"),  # Global-only model overrides any region
        (False, None, "us-central1"),  # Non-global model defaults to us-central1
        (
            False,
            "europe-west1",
            "europe-west1",
        ),  # Non-global model uses specified region
        (False, "us-east1", "us-east1"),  # Non-global model uses specified region
    ],
)
def test_get_vertex_region_global_only_model(
    model_is_global_only, vertex_region, expected_region
):
    """Test get_vertex_region ensures global-only models default to 'global' region"""
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    vertex_base = VertexBase()

    with patch(
        "litellm.llms.vertex_ai.vertex_llm_base.is_global_only_vertex_model"
    ) as mock_is_global_only:
        mock_is_global_only.return_value = model_is_global_only

        result = vertex_base.get_vertex_region(
            vertex_region=vertex_region, model="test-model"
        )

        assert result == expected_region
        mock_is_global_only.assert_called_once_with("test-model")


def test_vertex_filter_format_uri():
    import json

    from litellm.llms.vertex_ai.common_utils import filter_schema_fields

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "format": "uri",
                "description": "The URL to fetch content from",
            },
            "prompt": {
                "type": "string",
                "description": "The prompt to run on the fetched content",
            },
        },
        "required": ["url", "prompt"],
        "$schema": "http://json-schema.org/draft-07/schema#",
    }
    valid_schema_fields = {
        "minLength",
        "nullable",
        "maxItems",
        "required",
        "default",
        "items",
        "propertyOrdering",
        "maximum",
        "properties",
        "anyOf",
        "description",
        "minProperties",
        "minimum",
        "minItems",
        "maxProperties",
        "title",
        "pattern",
        "example",
        "format",
        "enum",
        "maxLength",
        "type",
    }

    new_parameters = filter_schema_fields(
        schema_dict=parameters,
        valid_fields=valid_schema_fields,
    )

    assert "uri" not in json.dumps(new_parameters)

def test_convert_schema_types_type_array_conversion():
    """
    Test _convert_schema_types function handles type arrays and case conversion.
    
    This test verifies the fix for the issue where type arrays like ["string", "number"] 
    would raise an exception in Vertex AI schema validation.

    Relevant issue: https://github.com/BerriAI/litellm/issues/14091
    """
    from litellm.llms.vertex_ai.common_utils import _convert_schema_types

    # Input: OpenAI-style schema with type array (the problematic case)
    input_schema = {
        "type": "object",
        "properties": {
            "studio": {
                "type": ["string", "number"],
                "description": "The studio ID or name"
            }
        },
        "required": ["studio"],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
    }

    # Expected output: Vertex AI compatible schema with anyOf and uppercase types
    expected_output = {
        "type": "object",
        "properties": {
            "studio": {
                "anyOf": [
                    {"type": "string"}, 
                    {"type": "number"}
                ],
                "description": "The studio ID or name"
            }
        },
        "required": ["studio"],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
    }

    # Apply the transformation
    _convert_schema_types(input_schema)

    # Verify the transformation
    assert input_schema == expected_output

    # Verify specific transformations:
    # 1. Root level type converted to uppercase
    assert input_schema["type"] == "object"

    # 2. Type array converted to anyOf format
    assert "anyOf" in input_schema["properties"]["studio"]
    assert "type" not in input_schema["properties"]["studio"]

    # 3. Individual types in anyOf are uppercase
    anyof_types = input_schema["properties"]["studio"]["anyOf"]
    assert anyof_types[0]["type"] == "string"
    assert anyof_types[1]["type"] == "number"

    # 4. Other properties preserved
    assert input_schema["properties"]["studio"]["description"] == "The studio ID or name"
    assert input_schema["required"] == ["studio"]


def test_fix_enum_empty_strings():
    """
    Test _fix_enum_empty_strings function replaces empty strings with None in enum arrays.
    
    This test verifies the fix for the issue where Gemini rejects tool definitions 
    with empty strings in enum values, causing API failures.

    Relevant issue: Gemini does not accept empty strings in enum values
    """
    from litellm.llms.vertex_ai.common_utils import _fix_enum_empty_strings

    # Input: Schema with empty string in enum (the problematic case)
    input_schema = {
        "type": "object",
        "properties": {
            "user_agent_type": {
                "enum": ["", "desktop", "mobile", "tablet"],
                "type": "string",
                "description": "Device type for user agent"
            }
        },
        "required": ["user_agent_type"]
    }

    # Expected output: Empty strings replaced with None
    expected_output = {
        "type": "object", 
        "properties": {
            "user_agent_type": {
                "enum": [None, "desktop", "mobile", "tablet"],
                "type": "string",
                "description": "Device type for user agent"
            }
        },
        "required": ["user_agent_type"]
    }

    # Apply the transformation
    _fix_enum_empty_strings(input_schema)

    # Verify the transformation
    assert input_schema == expected_output

    # Verify specific transformations:
    # 1. Empty string replaced with None
    enum_values = input_schema["properties"]["user_agent_type"]["enum"]
    assert "" not in enum_values
    assert None in enum_values

    # 2. Other enum values preserved
    assert "desktop" in enum_values
    assert "mobile" in enum_values
    assert "tablet" in enum_values

    # 3. Other properties preserved
    assert input_schema["properties"]["user_agent_type"]["type"] == "string"
    assert input_schema["properties"]["user_agent_type"]["description"] == "Device type for user agent"


def test_fix_enum_types():
    """
    Test _fix_enum_types function removes enum fields when type is not string.
    
    This test verifies the fix for the issue where Gemini rejects cached content
    with function parameter enums on non-string types, causing API failures.

    Relevant issue: Gemini only allows enums for string-typed fields
    """
    from litellm.llms.vertex_ai.common_utils import _fix_enum_types

    # Input: Schema with enum on non-string type (the problematic case)
    input_schema = {
        "type": "object",
        "properties": {
            "truncateMode": {
                "enum": ["auto", "none", "start", "end"],
                "type": "string",  # This should keep the enum
                "description": "How to truncate content"
            },
            "maxLength": {
                "enum": [100, 200, 500],  # This should be removed
                "type": "integer",
                "description": "Maximum length"
            },
            "enabled": {
                "enum": [True, False],  # This should be removed
                "type": "boolean",
                "description": "Whether feature is enabled"
            },
            "nested": {
                "type": "object",
                "properties": {
                    "innerEnum": {
                        "enum": ["a", "b", "c"],  # This should be kept
                        "type": "string"
                    },
                    "innerNonStringEnum": {
                        "enum": [1, 2, 3],  # This should be removed
                        "type": "integer"
                    }
                }
            },
            "anyOfField": {
                "anyOf": [
                    {"type": "string", "enum": ["option1", "option2"]},  # This should be kept
                    {"type": "integer", "enum": [1, 2, 3]}  # This should be removed
                ]
            }
        }
    }

    # Expected output: Non-string enums removed, string enums kept
    expected_output = {
        "type": "object",
        "properties": {
            "truncateMode": {
                "enum": ["auto", "none", "start", "end"],  # Kept - string type
                "type": "string",
                "description": "How to truncate content"
            },
            "maxLength": {  # enum removed
                "type": "integer",
                "description": "Maximum length"
            },
            "enabled": {  # enum removed
                "type": "boolean",
                "description": "Whether feature is enabled"
            },
            "nested": {
                "type": "object",
                "properties": {
                    "innerEnum": {
                        "enum": ["a", "b", "c"],  # Kept - string type
                        "type": "string"
                    },
                    "innerNonStringEnum": {  # enum removed
                        "type": "integer"
                    }
                }
            },
            "anyOfField": {
                "anyOf": [
                    {"type": "string", "enum": ["option1", "option2"]},  # Kept - has string type
                    {"type": "integer"}  # enum removed
                ]
            }
        }
    }

    # Apply the transformation
    _fix_enum_types(input_schema)

    # Verify the transformation
    assert input_schema == expected_output

    # Verify specific transformations:
    # 1. String enums are preserved
    assert "enum" in input_schema["properties"]["truncateMode"]
    assert input_schema["properties"]["truncateMode"]["enum"] == ["auto", "none", "start", "end"]
    
    assert "enum" in input_schema["properties"]["nested"]["properties"]["innerEnum"]
    assert input_schema["properties"]["nested"]["properties"]["innerEnum"]["enum"] == ["a", "b", "c"]

    # 2. Non-string enums are removed
    assert "enum" not in input_schema["properties"]["maxLength"]
    assert "enum" not in input_schema["properties"]["enabled"]
    assert "enum" not in input_schema["properties"]["nested"]["properties"]["innerNonStringEnum"]

    # 3. anyOf with string type keeps enum, non-string removes it
    assert "enum" in input_schema["properties"]["anyOfField"]["anyOf"][0]
    assert "enum" not in input_schema["properties"]["anyOfField"]["anyOf"][1]

    # 4. Other properties preserved
    assert input_schema["properties"]["maxLength"]["type"] == "integer"
    assert input_schema["properties"]["enabled"]["type"] == "boolean"


def test_get_token_url():
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexLLM,
    )

    vertex_llm = VertexLLM()
    vertex_ai_project = "pathrise-convert-1606954137718"
    vertex_ai_location = "us-central1"
    vertex_credentials = ""

    should_use_v1beta1_features = vertex_llm.is_using_v1beta1_features(
        optional_params={"cached_content": "hi"}
    )

    _, url = vertex_llm._get_token_and_url(
        auth_header=None,
        vertex_project=vertex_ai_project,
        vertex_location=vertex_ai_location,
        vertex_credentials=vertex_credentials,
        gemini_api_key="",
        custom_llm_provider="vertex_ai_beta",
        should_use_v1beta1_features=should_use_v1beta1_features,
        api_base=None,
        model="",
        stream=False,
    )

    print("url=", url)



    should_use_v1beta1_features = vertex_llm.is_using_v1beta1_features(
        optional_params={"temperature": 0.1}
    )

    _, url = vertex_llm._get_token_and_url(
        auth_header=None,
        vertex_project=vertex_ai_project,
        vertex_location=vertex_ai_location,
        vertex_credentials=vertex_credentials,
        gemini_api_key="",
        custom_llm_provider="vertex_ai_beta",
        should_use_v1beta1_features=should_use_v1beta1_features,
        api_base=None,
        model="",
        stream=False,
    )

    print("url for normal request", url)

    assert "v1beta1" not in url
    assert "/v1/" in url

    pass


@pytest.mark.asyncio
async def test_vertex_ai_token_counter_routes_partner_models():
    """
    Test that VertexAITokenCounter correctly routes partner models (Claude, Mistral, etc.)
    to the partner models token counter instead of the Gemini token counter.
    """
    from unittest.mock import AsyncMock, patch
    from litellm.llms.vertex_ai.common_utils import VertexAITokenCounter
    from litellm.types.utils import TokenCountResponse

    token_counter = VertexAITokenCounter()

    # Mock the partner models handler
    with patch(
        "litellm.llms.vertex_ai.vertex_ai_partner_models.main.VertexAIPartnerModels.count_tokens"
    ) as mock_partner_count_tokens:
        mock_partner_count_tokens.return_value = {
            "input_tokens": 42,
            "tokenizer_used": "vertex_ai_partner_models",
        }

        # Test with a Claude model (partner model)
        result = await token_counter.count_tokens(
            model_to_use="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={
                "litellm_params": {
                    "vertex_project": "test-project",
                    "vertex_location": "us-east5",
                }
            },
            request_model="vertex_ai/claude-3-5-sonnet-20241022",
        )

        # Verify partner models handler was called
        assert mock_partner_count_tokens.called
        assert result is not None
        assert isinstance(result, TokenCountResponse)
        assert result.total_tokens == 42
        assert result.tokenizer_type == "vertex_ai_partner_models"


@pytest.mark.asyncio
async def test_vertex_ai_token_counter_routes_gemini_models():
    """
    Test that VertexAITokenCounter correctly routes Gemini models
    to the Gemini token counter (not partner models).
    """
    from unittest.mock import AsyncMock, patch
    from litellm.llms.vertex_ai.common_utils import VertexAITokenCounter
    from litellm.types.utils import TokenCountResponse

    token_counter = VertexAITokenCounter()

    # Mock the Gemini handler (different import path)
    with patch(
        "litellm.llms.vertex_ai.count_tokens.handler.VertexAITokenCounter.acount_tokens"
    ) as mock_gemini_count_tokens:
        mock_gemini_count_tokens.return_value = {
            "totalTokens": 50,
            "tokenizer_used": "gemini",
        }

        # Test with a Gemini model (not a partner model)
        result = await token_counter.count_tokens(
            model_to_use="gemini-1.5-pro",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={
                "litellm_params": {
                    "vertex_project": "test-project",
                    "vertex_location": "us-central1",
                }
            },
            request_model="vertex_ai/gemini-1.5-pro",
        )

        # Verify Gemini handler was called
        assert mock_gemini_count_tokens.called
        assert result is not None
        assert isinstance(result, TokenCountResponse)
        assert result.total_tokens == 50


@pytest.mark.asyncio
async def test_vertex_ai_partner_model_detection():
    """
    Test that VertexAIPartnerModels.is_vertex_partner_model correctly identifies
    partner models (Claude, Mistral, Llama, etc.).
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
        VertexAIPartnerModels,
    )

    # Test Claude models (should be detected as partner model)
    assert VertexAIPartnerModels.is_vertex_partner_model("claude-3-5-sonnet-20241022")
    assert VertexAIPartnerModels.is_vertex_partner_model("claude-3-opus-20240229")
    assert VertexAIPartnerModels.is_vertex_partner_model("claude-3-haiku-20240307")

    # Test Mistral models
    assert VertexAIPartnerModels.is_vertex_partner_model("mistral-large-2407")
    assert VertexAIPartnerModels.is_vertex_partner_model("mistral-7b-instruct-v0.3")

    # Test Meta/Llama models
    assert VertexAIPartnerModels.is_vertex_partner_model("meta/llama-3.1-405b")
    # Test Minimax models
    assert VertexAIPartnerModels.is_vertex_partner_model("minimaxai/minimax-m2-maas")
    # Test Moonshot models
    assert VertexAIPartnerModels.is_vertex_partner_model(
        "moonshotai/kimi-k2-thinking-maas"
    )

    # Test Gemini models (should NOT be detected as partner model)
    assert not VertexAIPartnerModels.is_vertex_partner_model("gemini-1.5-pro")
    assert not VertexAIPartnerModels.is_vertex_partner_model("gemini-1.0-pro")
    assert not VertexAIPartnerModels.is_vertex_partner_model("gemini-pro-vision")

    # Test other non-partner models
    assert not VertexAIPartnerModels.is_vertex_partner_model("text-bison-001")
    assert not VertexAIPartnerModels.is_vertex_partner_model("chat-bison-001")


def test_vertex_ai_minimax_uses_openai_handler():
    """
    Ensure Minimax partner models re-use the OpenAI-format handler.
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
        VertexAIPartnerModels,
    )

    assert VertexAIPartnerModels.should_use_openai_handler(
        "minimaxai/minimax-m2-maas"
    )


def test_vertex_ai_moonshot_uses_openai_handler():
    """
    Ensure Moonshot partner models re-use the OpenAI-format handler.
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
        VertexAIPartnerModels,
    )

    assert VertexAIPartnerModels.should_use_openai_handler(
        "moonshotai/kimi-k2-thinking-maas"
    )
