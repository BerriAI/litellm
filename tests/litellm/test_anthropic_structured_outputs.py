"""
Tests for Anthropic structured outputs with constraint filtering.

These tests verify that LiteLLM properly filters unsupported JSON Schema
constraints before sending to Anthropic API, matching the behavior of the
official Anthropic SDK.

Test-driven development: These tests will FAIL initially until the fix is implemented.
"""

import json
import os
import sys
from typing import Dict, Any
import pytest
from pydantic import BaseModel, Field

# Set environment variable before litellm import for local testing
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestSchemaFiltering:
    """Test schema filtering removes unsupported constraints"""

    def test_filter_numerical_constraints(self):
        """Test that numerical constraints are removed"""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                },
                "price": {
                    "type": "number",
                    "minimum": 0.01,
                    "exclusiveMaximum": 1000000,
                    "multipleOf": 0.01,
                },
            },
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)

        # Verify numerical constraints are removed
        assert "minimum" not in filtered["properties"]["age"]
        assert "maximum" not in filtered["properties"]["age"]
        assert "minimum" not in filtered["properties"]["price"]
        assert "exclusiveMaximum" not in filtered["properties"]["price"]
        assert "multipleOf" not in filtered["properties"]["price"]

        # Verify type is preserved
        assert filtered["properties"]["age"]["type"] == "integer"
        assert filtered["properties"]["price"]["type"] == "number"

    def test_filter_string_constraints(self):
        """Test that string constraints are removed"""
        schema = {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 20,
                },
                "email": {
                    "type": "string",
                    "minLength": 5,
                    "maxLength": 255,
                },
            },
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)

        # Verify string constraints are removed
        assert "minLength" not in filtered["properties"]["username"]
        assert "maxLength" not in filtered["properties"]["username"]
        assert "minLength" not in filtered["properties"]["email"]
        assert "maxLength" not in filtered["properties"]["email"]

        # Verify type is preserved
        assert filtered["properties"]["username"]["type"] == "string"

    def test_filter_array_constraints(self):
        """Test that array constraints are handled correctly"""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,  # Should be preserved (0 or 1)
                    "maxItems": 10,  # Should be removed
                },
                "optional_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,  # Should be preserved (0 or 1)
                    "maxItems": 5,  # Should be removed
                },
                "required_items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 5,  # Should be removed (>1)
                    "maxItems": 20,  # Should be removed
                },
            },
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)

        # minItems with values 0 or 1 should be preserved
        assert filtered["properties"]["tags"]["minItems"] == 1
        assert filtered["properties"]["optional_tags"]["minItems"] == 0

        # maxItems should always be removed
        assert "maxItems" not in filtered["properties"]["tags"]
        assert "maxItems" not in filtered["properties"]["optional_tags"]
        assert "maxItems" not in filtered["properties"]["required_items"]

        # minItems > 1 should be removed
        assert "minItems" not in filtered["properties"]["required_items"]

    def test_filter_nested_schemas(self):
        """Test that nested object schemas are filtered recursively"""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "minimum": 18, "maximum": 65},
                        "name": {"type": "string", "minLength": 1, "maxLength": 100},
                    },
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)

        # Verify nested constraints are removed
        assert "minimum" not in filtered["properties"]["user"]["properties"]["age"]
        assert "maximum" not in filtered["properties"]["user"]["properties"]["age"]
        assert "minLength" not in filtered["properties"]["user"]["properties"]["name"]
        assert "maxLength" not in filtered["properties"]["user"]["properties"]["name"]
        assert "minimum" not in filtered["properties"]["metadata"]["properties"]["score"]
        assert "maximum" not in filtered["properties"]["metadata"]["properties"]["score"]

    def test_filter_complex_schemas_with_defs(self):
        """Test that schemas with $defs are filtered recursively"""
        schema = {
            "type": "object",
            "$defs": {
                "Age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                },
                "Username": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 20,
                },
            },
            "properties": {
                "age": {"$ref": "#/$defs/Age"},
                "username": {"$ref": "#/$defs/Username"},
            },
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)

        # Verify $defs constraints are removed
        assert "minimum" not in filtered["$defs"]["Age"]
        assert "maximum" not in filtered["$defs"]["Age"]
        assert "minLength" not in filtered["$defs"]["Username"]
        assert "maxLength" not in filtered["$defs"]["Username"]

        # Verify $refs are preserved
        assert filtered["properties"]["age"]["$ref"] == "#/$defs/Age"
        assert filtered["properties"]["username"]["$ref"] == "#/$defs/Username"

    def test_filter_schemas_with_anyof(self):
        """Test that schemas with anyOf are filtered recursively"""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0, "maximum": 100},
                        {"type": "string", "minLength": 1, "maxLength": 50},
                    ]
                }
            },
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)

        # Verify constraints in anyOf are removed
        assert "minimum" not in filtered["properties"]["value"]["anyOf"][0]
        assert "maximum" not in filtered["properties"]["value"]["anyOf"][0]
        assert "minLength" not in filtered["properties"]["value"]["anyOf"][1]
        assert "maxLength" not in filtered["properties"]["value"]["anyOf"][1]

    def test_filter_preserves_other_properties(self):
        """Test that non-constraint properties are preserved"""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "User's full name",
                    "title": "Name",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"],
                    "description": "User status",
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                    "default": 25,
                },
            },
            "required": ["name", "status"],
        }

        filtered = AnthropicConfig.filter_anthropic_output_schema(schema)

        # Verify constraints removed
        assert "minLength" not in filtered["properties"]["name"]
        assert "maxLength" not in filtered["properties"]["name"]
        assert "minimum" not in filtered["properties"]["age"]
        assert "maximum" not in filtered["properties"]["age"]

        # Verify other properties preserved
        assert filtered["properties"]["name"]["description"] == "User's full name"
        assert filtered["properties"]["name"]["title"] == "Name"
        assert filtered["properties"]["status"]["enum"] == ["active", "inactive"]
        assert filtered["properties"]["age"]["default"] == 25
        assert filtered["required"] == ["name", "status"]


class TestDescriptionUpdates:
    """Test that constraint information is added to descriptions"""

    def test_add_numerical_constraint_to_description(self):
        """Test adding numerical constraints to descriptions"""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "minimum": 18,
                    "maximum": 65,
                },
                "score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 100.0,
                },
            },
        }

        result = AnthropicConfig._add_constraint_to_descriptions(schema)

        # Check age description
        age_desc = result["properties"]["age"]["description"]
        assert "Must be at least 18" in age_desc
        assert "Must be at most 65" in age_desc

        # Check score description
        score_desc = result["properties"]["score"]["description"]
        assert "Must be at least 0.0" in score_desc
        assert "Must be at most 100.0" in score_desc

    def test_add_string_constraint_to_description(self):
        """Test adding string constraints to descriptions"""
        schema = {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 20,
                },
                "email": {
                    "type": "string",
                    "minLength": 5,
                    "maxLength": 255,
                },
            },
        }

        result = AnthropicConfig._add_constraint_to_descriptions(schema)

        # Check username description
        username_desc = result["properties"]["username"]["description"]
        assert "Minimum length: 3" in username_desc
        assert "Maximum length: 20" in username_desc

        # Check email description
        email_desc = result["properties"]["email"]["description"]
        assert "Minimum length: 5" in email_desc
        assert "Maximum length: 255" in email_desc

    def test_preserve_existing_description(self):
        """Test that existing descriptions are preserved and constraints appended"""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "minimum": 18,
                    "maximum": 65,
                    "description": "User's age",
                },
            },
        }

        result = AnthropicConfig._add_constraint_to_descriptions(schema)

        age_desc = result["properties"]["age"]["description"]
        # Original description should be preserved
        assert "User's age" in age_desc
        # Constraints should be added
        assert "Must be at least 18" in age_desc
        assert "Must be at most 65" in age_desc

    def test_add_constraints_to_nested_schemas(self):
        """Test adding constraints to nested object schemas"""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "minimum": 0, "maximum": 150},
                        "name": {"type": "string", "minLength": 1, "maxLength": 100},
                    },
                },
            },
        }

        result = AnthropicConfig._add_constraint_to_descriptions(schema)

        # Check nested descriptions
        age_desc = result["properties"]["user"]["properties"]["age"]["description"]
        assert "Must be at least 0" in age_desc
        assert "Must be at most 150" in age_desc

        name_desc = result["properties"]["user"]["properties"]["name"]["description"]
        assert "Minimum length: 1" in name_desc
        assert "Maximum length: 100" in name_desc


class TestResponseValidation:
    """Test response validation against original schema"""

    def test_validate_response_valid(self):
        """Test that valid responses pass validation"""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
            },
            "required": ["age", "name"],
        }

        response = {"age": 25, "name": "John"}

        # Should not raise
        AnthropicConfig._validate_response_against_schema(
            response, schema, "claude-sonnet-4-5"
        )

    def test_validate_response_constraint_violation_maximum(self):
        """Test that maximum constraint violations are caught"""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            },
        }

        response = {"age": 200}  # Violates maximum

        with pytest.raises(ValueError, match="violates the original schema constraints"):
            AnthropicConfig._validate_response_against_schema(
                response, schema, "claude-sonnet-4-5"
            )

    def test_validate_response_constraint_violation_minimum(self):
        """Test that minimum constraint violations are caught"""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            },
        }

        response = {"age": -5}  # Violates minimum

        with pytest.raises(ValueError, match="violates the original schema constraints"):
            AnthropicConfig._validate_response_against_schema(
                response, schema, "claude-sonnet-4-5"
            )

    def test_validate_response_string_length_violation(self):
        """Test that string length constraint violations are caught"""
        schema = {
            "type": "object",
            "properties": {
                "username": {"type": "string", "minLength": 3, "maxLength": 20}
            },
        }

        # Too short
        response_short = {"username": "ab"}
        with pytest.raises(ValueError, match="violates the original schema constraints"):
            AnthropicConfig._validate_response_against_schema(
                response_short, schema, "claude-sonnet-4-5"
            )

        # Too long
        response_long = {"username": "a" * 21}
        with pytest.raises(ValueError, match="violates the original schema constraints"):
            AnthropicConfig._validate_response_against_schema(
                response_long, schema, "claude-sonnet-4-5"
            )

    def test_validate_response_missing_required_field(self):
        """Test that missing required fields are caught"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        response = {"name": "John"}  # Missing age

        with pytest.raises(ValueError, match="violates the original schema constraints"):
            AnthropicConfig._validate_response_against_schema(
                response, schema, "claude-sonnet-4-5"
            )

    def test_validate_response_nested_schema(self):
        """Test validation of nested schemas"""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "minimum": 18, "maximum": 65}
                    },
                }
            },
        }

        # Valid nested response
        valid_response = {"user": {"age": 25}}
        AnthropicConfig._validate_response_against_schema(
            valid_response, schema, "claude-sonnet-4-5"
        )

        # Invalid nested response
        invalid_response = {"user": {"age": 200}}
        with pytest.raises(ValueError, match="violates the original schema constraints"):
            AnthropicConfig._validate_response_against_schema(
                invalid_response, schema, "claude-sonnet-4-5"
            )


class TestIntegration:
    """Integration tests with actual litellm.completion calls"""

    @pytest.mark.asyncio
    async def test_anthropic_completion_with_numerical_constraints(self):
        """Test that schemas with numerical constraints work end-to-end"""
        # Skip if no API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        response = await litellm.acompletion(
            model="claude-sonnet-4-5-20250929",
            messages=[
                {
                    "role": "user",
                    "content": "Generate person info: name is John, age is 25",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "age": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 150,
                            },
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 100,
                            },
                        },
                        "required": ["age", "name"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        # Should not raise 400 error
        assert response.choices[0].message.content

        # Verify response is valid JSON
        result = json.loads(response.choices[0].message.content)
        assert "name" in result
        assert "age" in result

    @pytest.mark.asyncio
    async def test_anthropic_completion_with_string_constraints(self):
        """Test that schemas with string constraints work end-to-end"""
        # Skip if no API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        response = await litellm.acompletion(
            model="claude-sonnet-4-5-20250929",
            messages=[
                {
                    "role": "user",
                    "content": "Generate user data: username is john_doe, email is john@example.com",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "user",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "minLength": 3,
                                "maxLength": 20,
                            },
                            "email": {
                                "type": "string",
                                "minLength": 5,
                                "maxLength": 255,
                            },
                        },
                        "required": ["username", "email"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        # Should not raise 400 error
        assert response.choices[0].message.content

        # Verify response is valid JSON
        result = json.loads(response.choices[0].message.content)
        assert "username" in result
        assert "email" in result

    @pytest.mark.asyncio
    async def test_anthropic_completion_with_pydantic_model(self):
        """Test that Pydantic models with constraints work end-to-end"""
        # Skip if no API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        class PersonInfo(BaseModel):
            name: str = Field(min_length=1, max_length=100)
            age: int = Field(ge=0, le=150)
            email: str = Field(min_length=5, max_length=255)

        schema = PersonInfo.model_json_schema()

        response = await litellm.acompletion(
            model="claude-sonnet-4-5-20250929",
            messages=[
                {
                    "role": "user",
                    "content": "Generate person: name John, age 25, email john@example.com",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person_info", "schema": schema},
            },
        )

        # Should not raise 400 error
        assert response.choices[0].message.content

        # Verify response is valid JSON
        result = json.loads(response.choices[0].message.content)
        assert "name" in result
        assert "age" in result
        assert "email" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
