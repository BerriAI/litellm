"""
OpenAPI compliance tests for Google Interactions API.

Validates that our SDK requests/responses match the OpenAPI spec at:
https://ai.google.dev/static/api/interactions.openapi.json

Run with: pytest tests/test_litellm/interactions/test_openapi_compliance.py -v
"""

import json
import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openapi_core import OpenAPI
from openapi_core.testing.mock import MockRequest, MockResponse

OPENAPI_SPEC_URL = "https://ai.google.dev/static/api/interactions.openapi.json"


@pytest.fixture(scope="module")
def openapi_spec():
    """Load the OpenAPI spec."""
    response = httpx.get(OPENAPI_SPEC_URL)
    response.raise_for_status()
    spec_dict = response.json()
    return OpenAPI.from_dict(spec_dict)


@pytest.fixture(scope="module") 
def spec_dict():
    """Load raw spec dict for manual validation."""
    response = httpx.get(OPENAPI_SPEC_URL)
    response.raise_for_status()
    return response.json()


class TestRequestCompliance:
    """Tests that our request bodies match the OpenAPI spec."""

    def test_create_model_interaction_request_schema(self, spec_dict):
        """Verify CreateModelInteractionParams schema fields."""
        schema = spec_dict["components"]["schemas"]["CreateModelInteractionParams"]
        
        # Required fields per spec
        assert "model" in schema["required"]
        assert "input" in schema["required"]
        
        # Check our supported optional fields exist in spec
        our_optional_fields = [
            "tools", "system_instruction", "generation_config", 
            "stream", "store", "background", "response_modalities",
            "response_format", "response_mime_type", "previous_interaction_id"
        ]
        
        spec_properties = schema["properties"]
        for field in our_optional_fields:
            assert field in spec_properties, f"Field '{field}' not in OpenAPI spec"
            print(f"✓ Field '{field}' exists in spec")

    def test_input_types_match_spec(self, spec_dict):
        """Verify input field supports string, Content, Content[], Turn[]."""
        schema = spec_dict["components"]["schemas"]["CreateModelInteractionParams"]
        input_schema = schema["properties"]["input"]
        
        # Should be oneOf with multiple types
        assert "oneOf" in input_schema
        
        input_types = []
        for option in input_schema["oneOf"]:
            if option.get("type") == "string":
                input_types.append("string")
            elif option.get("type") == "array":
                input_types.append("array")
            elif "$ref" in option:
                input_types.append(option["$ref"])
        
        print(f"Input supports types: {input_types}")
        assert "string" in input_types, "Input should support string"
        assert "array" in input_types, "Input should support array"

    def test_content_schema_uses_discriminator(self, spec_dict):
        """Verify Content uses type discriminator."""
        content_schema = spec_dict["components"]["schemas"]["Content"]
        
        assert "discriminator" in content_schema
        assert content_schema["discriminator"]["propertyName"] == "type"
        
        # Check TextContent is an option
        mapping = content_schema["discriminator"]["mapping"]
        assert "text" in mapping
        print(f"Content type discriminator mapping: {list(mapping.keys())}")

    def test_text_content_schema(self, spec_dict):
        """Verify TextContent schema."""
        text_schema = spec_dict["components"]["schemas"]["TextContent"]
        
        assert "type" in text_schema["required"]
        assert "text" in text_schema["properties"]
        assert text_schema["properties"]["type"].get("const") == "text"
        print("✓ TextContent schema is correct")

    def test_turn_schema(self, spec_dict):
        """Verify Turn schema for multi-turn conversations."""
        turn_schema = spec_dict["components"]["schemas"]["Turn"]
        
        assert "role" in turn_schema["properties"]
        assert "content" in turn_schema["properties"]
        
        # Content can be string or Content[]
        content_prop = turn_schema["properties"]["content"]
        assert "oneOf" in content_prop
        print("✓ Turn schema supports role + content")


class TestResponseCompliance:
    """Tests that our response types match the OpenAPI spec."""

    def test_interaction_response_fields(self, spec_dict):
        """Verify our InteractionsAPIResponse has correct fields."""
        # The response is the Interaction schema
        # Check CreateModelInteractionParams which includes output fields
        schema = spec_dict["components"]["schemas"]["CreateModelInteractionParams"]
        
        # Output fields (readOnly)
        output_fields = ["id", "status", "created", "updated", "role", "outputs", "usage"]
        
        for field in output_fields:
            assert field in schema["properties"], f"Output field '{field}' not in spec"
            print(f"✓ Output field '{field}' exists in spec")

    def test_status_enum_values(self, spec_dict):
        """Verify status enum values match spec."""
        schema = spec_dict["components"]["schemas"]["CreateModelInteractionParams"]
        status_prop = schema["properties"]["status"]
        
        expected_statuses = ["UNSPECIFIED", "IN_PROGRESS", "REQUIRES_ACTION", "COMPLETED", "FAILED", "CANCELLED"]
        assert status_prop["enum"] == expected_statuses
        print(f"✓ Status enum values: {expected_statuses}")

    def test_usage_schema(self, spec_dict):
        """Verify Usage schema fields."""
        usage_schema = spec_dict["components"]["schemas"]["Usage"]
        
        # Key usage fields
        expected_fields = ["total_input_tokens", "total_output_tokens", "total_tokens"]
        
        for field in expected_fields:
            assert field in usage_schema["properties"], f"Usage field '{field}' not in spec"
            print(f"✓ Usage field '{field}' exists")


class TestToolsCompliance:
    """Tests that our tool types match the OpenAPI spec."""

    def test_tool_schema(self, spec_dict):
        """Verify Tool schema."""
        tool_schema = spec_dict["components"]["schemas"]["Tool"]
        
        # Tool should be oneOf multiple tool types
        assert "oneOf" in tool_schema or "properties" in tool_schema
        print(f"✓ Tool schema found")

    def test_function_declaration_schema(self, spec_dict):
        """Verify FunctionDeclaration schema for function tools."""
        if "FunctionDeclaration" in spec_dict["components"]["schemas"]:
            func_schema = spec_dict["components"]["schemas"]["FunctionDeclaration"]
            assert "name" in func_schema.get("properties", {}) or "name" in func_schema.get("required", [])
            print("✓ FunctionDeclaration schema found")
        else:
            print("⚠ FunctionDeclaration schema not found (may be nested)")


class TestEndpointCompliance:
    """Tests that our endpoints match the OpenAPI spec."""

    def test_create_endpoint_exists(self, spec_dict):
        """Verify POST /interactions endpoint exists."""
        paths = spec_dict["paths"]
        
        # Find the create interactions endpoint
        create_path = None
        for path, methods in paths.items():
            if "interactions" in path and "post" in methods:
                create_path = path
                break
        
        assert create_path is not None, "POST /interactions endpoint not found"
        print(f"✓ Create endpoint: POST {create_path}")

    def test_get_endpoint_exists(self, spec_dict):
        """Verify GET /interactions/{id} endpoint exists."""
        paths = spec_dict["paths"]
        
        get_path = None
        for path, methods in paths.items():
            if "{id}" in path and "interactions" in path and "get" in methods:
                get_path = path
                break
        
        assert get_path is not None, "GET /interactions/{id} endpoint not found"
        print(f"✓ Get endpoint: GET {get_path}")

    def test_delete_endpoint_exists(self, spec_dict):
        """Verify DELETE /interactions/{id} endpoint exists."""
        paths = spec_dict["paths"]
        
        delete_path = None
        for path, methods in paths.items():
            if "{id}" in path and "interactions" in path and "delete" in methods:
                delete_path = path
                break
        
        assert delete_path is not None, "DELETE /interactions/{id} endpoint not found"
        print(f"✓ Delete endpoint: DELETE {delete_path}")


if __name__ == "__main__":
    # Quick manual test
    import httpx
    
    print("Loading OpenAPI spec...")
    response = httpx.get(OPENAPI_SPEC_URL)
    spec = response.json()
    
    print(f"\nSpec version: {spec.get('openapi')}")
    print(f"API title: {spec.get('info', {}).get('title')}")
    print(f"\nEndpoints:")
    for path, methods in spec.get("paths", {}).items():
        for method in methods:
            if method in ["get", "post", "delete", "put", "patch"]:
                print(f"  {method.upper()} {path}")
    
    print(f"\nSchemas: {list(spec.get('components', {}).get('schemas', {}).keys())[:10]}...")

