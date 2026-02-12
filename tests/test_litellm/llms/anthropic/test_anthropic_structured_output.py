"""
Anthropic structured outputs don't support these constraints:
- Numerical: minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf
- String: minLength, maxLength, pattern
- Array: maxItems, minItems (except 0/1)

We filter them out, add constraint info to descriptions, and validate responses.
Ref: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
"""

import pytest
import json
from pydantic import BaseModel, Field
from typing import List


class TestAnthropicStructuredOutput:
    """Test Anthropic structured output schema transformations."""

    def test_max_length_on_list_field_filtered(self):
        """
        Test that max_length on List fields is filtered out for Anthropic models.
        
        Anthropic doesn't support 'maxItems' property for array types in their
        output_format.schema, so we need to filter it out.
        
        Related issue: https://github.com/BerriAI/litellm/issues/19444
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        
        # Define a Pydantic model with max_length on a List field
        class ResponseModel(BaseModel):
            items: List[str] = Field(max_length=5, description="List of items")
            name: str = Field(description="Name field")
        
        config = AnthropicConfig()
        
        # Get the JSON schema from the Pydantic model
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)
        
        # Extract the actual schema
        schema = json_schema["json_schema"]["schema"]
        
        # Verify that maxItems is present in the raw schema (from Pydantic)
        assert "maxItems" in schema["properties"]["items"]
        
        # Now apply the Anthropic output format transformation
        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"]
        }
        
        output_format = config.map_response_format_to_anthropic_output_format(
            response_format
        )
        
        # Verify that maxItems is filtered out for Anthropic
        assert output_format is not None
        assert "schema" in output_format
        transformed_schema = output_format["schema"]
        
        # maxItems should be removed from the items property
        assert "maxItems" not in transformed_schema["properties"]["items"]
        
        # But other properties should remain
        assert "type" in transformed_schema["properties"]["items"]
        assert transformed_schema["properties"]["items"]["type"] == "array"
        assert "description" in transformed_schema["properties"]["items"]

    def test_min_length_on_list_field_filtered(self):
        """
        Test that min_length on List fields is filtered out for Anthropic models.
        
        Anthropic likely doesn't support 'minItems' either.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        
        class ResponseModel(BaseModel):
            items: List[str] = Field(min_length=2, description="List of items")
        
        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)
        
        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"]
        }
        
        output_format = config.map_response_format_to_anthropic_output_format(
            response_format
        )
        
        assert output_format is not None
        transformed_schema = output_format["schema"]
        
        # minItems should be removed
        assert "minItems" not in transformed_schema["properties"]["items"]

    def test_nested_array_constraints_filtered(self):
        """
        Test that array constraints are filtered at all nesting levels.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        
        class NestedItem(BaseModel):
            tags: List[str] = Field(max_length=3)
        
        class ResponseModel(BaseModel):
            items: List[NestedItem] = Field(max_length=5)
        
        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)
        
        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"]
        }
        
        output_format = config.map_response_format_to_anthropic_output_format(
            response_format
        )
        
        assert output_format is not None
        transformed_schema = output_format["schema"]
        
        # Top-level maxItems should be removed
        assert "maxItems" not in transformed_schema["properties"]["items"]
        
        # Nested maxItems should also be removed
        if "$defs" in transformed_schema:
            nested_item_schema = transformed_schema["$defs"].get("NestedItem", {})
            if "properties" in nested_item_schema and "tags" in nested_item_schema["properties"]:
                assert "maxItems" not in nested_item_schema["properties"]["tags"]

    def test_string_and_number_constraints_filtered(self):
        """
        Test that string/number constraints are filtered (not supported by Anthropic).
        Ref: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class ResponseModel(BaseModel):
            name: str = Field(max_length=100, min_length=1, description="Name")
            age: int = Field(ge=0, le=150, description="Age")
            items: List[str] = Field(description="Items list")

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)

        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"]
        }

        output_format = config.map_response_format_to_anthropic_output_format(
            response_format
        )

        assert output_format is not None
        transformed_schema = output_format["schema"]

        # String constraints should be FILTERED OUT
        name_schema = transformed_schema["properties"]["name"]
        assert "maxLength" not in name_schema
        assert "minLength" not in name_schema
        # But constraint info should be in description
        assert "Minimum length: 1" in name_schema.get("description", "")
        assert "Maximum length: 100" in name_schema.get("description", "")

        # Number constraints should be FILTERED OUT
        age_schema = transformed_schema["properties"]["age"]
        assert "minimum" not in age_schema
        assert "maximum" not in age_schema
        # But constraint info should be in description
        assert "Must be at least 0" in age_schema.get("description", "")
        assert "Must be at most 150" in age_schema.get("description", "")

    def test_pattern_filtered(self):
        """Test that pattern constraints are filtered (not supported by Anthropic)."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                }
            }
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)
        assert "pattern" not in filtered["properties"]["email"]

    def test_definitions_recursion(self):
        """Test that 'definitions' (not just $defs) are recursively filtered."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        schema = {
            "type": "object",
            "definitions": {
                "User": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "minimum": 0, "maximum": 150}
                    }
                }
            },
            "properties": {
                "user": {"$ref": "#/definitions/User"}
            }
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)
        assert "minimum" not in filtered["definitions"]["User"]["properties"]["age"]
        assert "maximum" not in filtered["definitions"]["User"]["properties"]["age"]

    def test_oneof_filtering(self):
        """Test oneOf schemas are recursively filtered."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "integer", "minimum": 0, "maximum": 100},
                        {"type": "string", "minLength": 1, "maxLength": 50}
                    ]
                }
            }
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)
        assert "minimum" not in filtered["properties"]["value"]["oneOf"][0]
        assert "maximum" not in filtered["properties"]["value"]["oneOf"][0]
        assert "minLength" not in filtered["properties"]["value"]["oneOf"][1]
        assert "maxLength" not in filtered["properties"]["value"]["oneOf"][1]

    def test_allof_filtering(self):
        """Test allOf schemas are recursively filtered."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "allOf": [
                        {"type": "integer"},
                        {"minimum": 0, "maximum": 100}
                    ]
                }
            }
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)
        assert "minimum" not in filtered["properties"]["value"]["allOf"][1]
        assert "maximum" not in filtered["properties"]["value"]["allOf"][1]

    def test_minitems_edge_cases(self):
        """Test minItems: 0 and 1 preserved, >1 removed."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        schema = {
            "type": "object",
            "properties": {
                "optional": {"type": "array", "items": {"type": "string"}, "minItems": 0},
                "required": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "many": {"type": "array", "items": {"type": "string"}, "minItems": 5}
            }
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)
        assert filtered["properties"]["optional"]["minItems"] == 0  # Preserved
        assert filtered["properties"]["required"]["minItems"] == 1  # Preserved
        assert "minItems" not in filtered["properties"]["many"]  # Removed

    def test_additional_properties_added(self):
        """Test additionalProperties: false is added to objects."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)
        assert filtered["additionalProperties"] is False

    def test_validation_error_message(self):
        """Test response validation error messages."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            }
        }

        response = {"age": 200}

        with pytest.raises(ValueError) as exc_info:
            AnthropicConfig._validate_response_against_schema(response, schema, "claude-sonnet-4-5")

        error_msg = str(exc_info.value)
        assert "violates the original schema constraints" in error_msg
        assert "age" in error_msg or "200" in error_msg
