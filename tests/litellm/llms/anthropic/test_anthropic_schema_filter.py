"""
Tests for Anthropic JSON schema filtering.

Related to: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#how-sdk-transformation-works
"""

import pytest
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestFilterAnthropicOutputSchema:
    """Test the filter_anthropic_output_schema function."""

    def test_removes_numeric_constraints(self):
        """Test that minimum/maximum are removed from numeric schemas."""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                    "description": "Person's age"
                },
                "score": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 100
                }
            }
        }
        
        result = AnthropicConfig.filter_anthropic_output_schema(schema)
        
        # minimum/maximum should be removed
        assert "minimum" not in result["properties"]["age"]
        assert "maximum" not in result["properties"]["age"]
        assert "exclusiveMinimum" not in result["properties"]["score"]
        assert "exclusiveMaximum" not in result["properties"]["score"]
        
        # Other fields preserved
        assert result["properties"]["age"]["type"] == "integer"
        # Description should be updated with removed constraint info
        assert "Person's age" in result["properties"]["age"]["description"]
        assert "minimum value: 0" in result["properties"]["age"]["description"]
        assert "maximum value: 150" in result["properties"]["age"]["description"]
        # Score had no description, should get one from constraints
        assert "exclusive minimum value: 0" in result["properties"]["score"]["description"]
        assert "exclusive maximum value: 100" in result["properties"]["score"]["description"]

    def test_removes_string_constraints(self):
        """Test that minLength/maxLength are removed from string schemas."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100
                }
            }
        }
        
        result = AnthropicConfig.filter_anthropic_output_schema(schema)
        
        assert "minLength" not in result["properties"]["name"]
        assert "maxLength" not in result["properties"]["name"]
        assert result["properties"]["name"]["type"] == "string"
        # Description should contain constraint info
        assert "minimum length: 1" in result["properties"]["name"]["description"]
        assert "maximum length: 100" in result["properties"]["name"]["description"]

    def test_removes_array_constraints(self):
        """Test that minItems/maxItems are removed from array schemas."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10
        }
        
        result = AnthropicConfig.filter_anthropic_output_schema(schema)
        
        assert "minItems" not in result
        assert "maxItems" not in result
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}
        # Description should contain constraint info
        assert "minimum number of items: 1" in result["description"]
        assert "maximum number of items: 10" in result["description"]

    def test_handles_nested_schemas(self):
        """Test that nested schemas are also filtered."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "quantity": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100
                            }
                        }
                    },
                    "minItems": 1
                }
            }
        }
        
        result = AnthropicConfig.filter_anthropic_output_schema(schema)
        
        # Top level array constraint removed
        assert "minItems" not in result["properties"]["items"]
        
        # Nested numeric constraints removed
        nested_props = result["properties"]["items"]["items"]["properties"]
        assert "minimum" not in nested_props["quantity"]
        assert "maximum" not in nested_props["quantity"]

    def test_handles_anyof_oneof_allof(self):
        """Test that anyOf/oneOf/allOf schemas are filtered recursively."""
        schema = {
            "anyOf": [
                {"type": "integer", "minimum": 0},
                {"type": "string", "minLength": 1}
            ]
        }
        
        result = AnthropicConfig.filter_anthropic_output_schema(schema)
        
        assert "minimum" not in result["anyOf"][0]
        assert "minLength" not in result["anyOf"][1]

    def test_preserves_valid_fields(self):
        """Test that valid schema fields are preserved."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"],
                    "description": "Current status"
                }
            },
            "required": ["status"],
            "additionalProperties": False
        }
        
        result = AnthropicConfig.filter_anthropic_output_schema(schema)
        
        assert result == schema  # Should be unchanged
