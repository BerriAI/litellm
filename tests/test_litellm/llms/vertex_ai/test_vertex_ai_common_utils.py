import os
import sys
from unittest.mock import patch

import pytest

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

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
        value=non_default_params["response_format"], optional_params=optional_params, model="gemini-1.5-pro-preview-0409"
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


def test_get_vertex_model_id_from_url():
    """Test get_vertex_model_id_from_url with various URLs"""
    from litellm.llms.vertex_ai.common_utils import get_vertex_model_id_from_url

    # Test with valid URL
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent"
    model_id = get_vertex_model_id_from_url(url)
    assert model_id == "gemini-pro"

    # Test with invalid URL
    url = "https://invalid-url.com"
    model_id = get_vertex_model_id_from_url(url)
    assert model_id is None


def test_get_vertex_model_id_from_url_with_slashes():
    """Test get_vertex_model_id_from_url with model names containing slashes (e.g., gcp/google/gemini-2.5-flash)

    Regression test for NVIDIA issue: custom model names with slashes in passthrough URLs
    were being truncated (e.g., 'gcp/google/gemini-2.5-flash' -> 'gcp'), causing access_groups
    checks to fail.
    """
    from litellm.llms.vertex_ai.common_utils import get_vertex_model_id_from_url

    # Test with model name containing slashes: gcp/google/gemini-2.5-flash
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gcp/google/gemini-2.5-flash:generateContent"
    model_id = get_vertex_model_id_from_url(url)
    assert model_id == "gcp/google/gemini-2.5-flash"

    # Test with model name containing slashes: gcp/google/gemini-3-flash-preview
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/gcp/google/gemini-3-flash-preview:streamGenerateContent"
    model_id = get_vertex_model_id_from_url(url)
    assert model_id == "gcp/google/gemini-3-flash-preview"

    # Test with custom model path: custom/model
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/custom/model:generateContent"
    model_id = get_vertex_model_id_from_url(url)
    assert model_id == "custom/model"

    # Test passthrough URL format (without host)
    url = "v1/projects/my-project/locations/us-central1/publishers/google/models/gcp/google/gemini-2.5-flash:generateContent"
    model_id = get_vertex_model_id_from_url(url)
    assert model_id == "gcp/google/gemini-2.5-flash"


def test_construct_target_url_with_version_prefix():
    """Test construct_target_url with version prefixes"""
    from litellm.llms.vertex_ai.common_utils import construct_target_url

    # Test with /v1/ prefix
    url = "/v1/publishers/google/models/gemini-pro:streamGenerateContent"
    vertex_project = "test-project"
    vertex_location = "us-central1"
    base_url = "https://us-central1-aiplatform.googleapis.com"

    target_url = construct_target_url(
        base_url=base_url,
        requested_route=url,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )

    expected_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent"
    assert str(target_url) == expected_url

    # Test with /v1beta1/ prefix
    url = "/v1beta1/publishers/google/models/gemini-pro:streamGenerateContent"

    target_url = construct_target_url(
        base_url=base_url,
        requested_route=url,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )

    expected_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent"
    assert str(target_url) == expected_url


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
                "description": "How to truncate content",
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
async def test_vertex_ai_token_counter_uses_count_tokens_location():
    """
    Test that VertexAITokenCounter uses vertex_count_tokens_location to override
    vertex_location when counting tokens for partner models.

    Count tokens API is not available on global location for partner models:
    https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/count-tokens
    """
    from unittest.mock import patch

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

        # Test with vertex_count_tokens_location overriding vertex_location
        await token_counter.count_tokens(
            model_to_use="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={
                "litellm_params": {
                    "vertex_project": "test-project",
                    "vertex_location": "global",  # Original location (not supported for count_tokens)
                    "vertex_count_tokens_location": "us-east5",  # Override for count_tokens
                }
            },
            request_model="vertex_ai/claude-3-5-sonnet-20241022",
        )

        # Verify the partner models handler was called with the overridden location
        assert mock_partner_count_tokens.called
        call_kwargs = mock_partner_count_tokens.call_args.kwargs
        assert call_kwargs["vertex_location"] == "us-east5"
        assert call_kwargs["vertex_project"] == "test-project"


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


def test_vertex_ai_zai_uses_openai_handler():
    """
    Ensure ZAI partner models re-use the OpenAI-format handler.
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
        VertexAIPartnerModels,
    )

    assert VertexAIPartnerModels.should_use_openai_handler(
        "zai-org/glm-4.7-maas"
    )


def test_vertex_ai_zai_is_partner_model():
    """
    Ensure ZAI models are detected as Vertex AI partner models.
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
        VertexAIPartnerModels,
    )

    assert VertexAIPartnerModels.is_vertex_partner_model("zai-org/glm-4.7-maas")


def test_build_vertex_schema_empty_properties():
    """
    Test _build_vertex_schema handles empty properties objects correctly.
    
    This test verifies the fix for the issue where Gemini rejects schemas 
    with empty properties objects like {"properties": {}, "type": "object"}.
    
    Error from Gemini: "GenerateContentRequest.generation_config.response_schema
    .properties[\"action\"].items.any_of[0].properties[\"go_back\"].properties: 
    should be non-empty for OBJECT type"
    
    The fix removes empty properties objects and their associated type/required fields.
    """
    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

    # Input: Schema with empty properties (the problematic case from real request)
    input_schema = {
        "properties": {
            "action": {
                "description": "List of actions to execute",
                "items": {
                    "anyOf": [
                        {
                            "properties": {
                                "go_back": {
                                    "properties": {},
                                    "type": "object",
                                    "additionalProperties": False,
                                    "description": "Go back",
                                    "required": []
                                }
                            },
                            "required": ["go_back"],
                            "type": "object",
                            "additionalProperties": False
                        }
                    ]
                },
                "type": "array"
            }
        },
        "type": "object",
        "additionalProperties": False
    }

    # Apply the transformation
    result = _build_vertex_schema(input_schema)

    # Verify the transformation removed empty properties
    # Navigate to the go_back schema
    go_back_schema = result["properties"]["action"]["items"]["anyOf"][0]["properties"]["go_back"]
    
    # Verify empty properties was removed
    assert "properties" not in go_back_schema, "Empty properties should be removed"
    
    # Verify type is kept as object (Gemini requires type: object even without properties)
    assert go_back_schema.get("type") == "object", "Type should be kept as object when properties is empty"
    
    # Verify required was also removed
    assert "required" not in go_back_schema, "Required should be removed when properties is empty"
    
    # Verify description is preserved
    assert go_back_schema.get("description") == "Go back", "Description should be preserved"
    
    # Verify parent schema still has proper structure
    parent_schema = result["properties"]["action"]["items"]["anyOf"][0]
    assert parent_schema["type"] == "object", "Parent schema should still have object type"
    assert "go_back" in parent_schema["properties"], "go_back should still be in parent properties"


def test_add_object_type_schema_with_no_properties_and_no_type():
    """
    Test that add_object_type adds type: object when schema has no properties and no type.
    Fixes issue where tools with no arguments (e.g. EnterPlanMode) fail on Gemini.
    """
    from litellm.llms.vertex_ai.common_utils import add_object_type

    # Input: Schema with no properties and no type (the problematic case)
    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema"
    }

    # Apply the transformation
    add_object_type(input_schema)

    # Verify type: object was added
    assert input_schema.get("type") == "object", "type: object should be added"

    # Verify $schema is preserved
    assert input_schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"


def test_add_object_type_does_not_override_existing_type():
    """
    Test add_object_type does not override existing type field.
    """
    from litellm.llms.vertex_ai.common_utils import add_object_type

    # Input: Schema with existing type
    input_schema = {
        "type": "string",
        "description": "A string field"
    }

    # Apply the transformation
    add_object_type(input_schema)

    # Verify type was not changed
    assert input_schema.get("type") == "string", "Existing type should not be changed"


def test_add_object_type_does_not_add_type_when_anyof_present():
    """
    Test add_object_type does not add type: object when anyOf is present.
    """
    from litellm.llms.vertex_ai.common_utils import add_object_type

    # Input: Schema with anyOf but no type
    input_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"}
        ]
    }

    # Apply the transformation
    add_object_type(input_schema)

    # Verify type was not added (anyOf handles the type)
    assert "type" not in input_schema, "type should not be added when anyOf is present"
