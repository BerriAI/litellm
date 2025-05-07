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
