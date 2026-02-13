"""
Test Anthropic structured output with Pydantic models.

This test file verifies that Pydantic models with various constraints
are properly converted to Anthropic-compatible JSON schemas.
"""

import pytest
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

    def test_string_constraints_preserved(self):
        """
        Test that string constraints (maxLength, minLength) are preserved
        since Anthropic supports these.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class ResponseModel(BaseModel):
            name: str = Field(max_length=100, min_length=1, description="Name")
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

        # String constraints should be preserved
        name_schema = transformed_schema["properties"]["name"]
        assert "maxLength" in name_schema
        assert "minLength" in name_schema
        assert name_schema["maxLength"] == 100
        assert name_schema["minLength"] == 1

    def test_numeric_ge_le_constraints_filtered(self):
        """
        Test that ge/le constraints (minimum/maximum) are filtered out.

        Pydantic's ge/le generate minimum/maximum in JSON schema, but Anthropic
        rejects these with validation errors.

        Related issue: https://github.com/BerriAI/litellm/issues/21016
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class ResponseModel(BaseModel):
            age: int = Field(ge=0, le=150, description="Age")
            score: float = Field(ge=0.0, le=1.0, description="Score")

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)

        # Verify Pydantic generates minimum/maximum in the raw schema
        raw_schema = json_schema["json_schema"]["schema"]
        assert "minimum" in raw_schema["properties"]["age"]
        assert "maximum" in raw_schema["properties"]["age"]

        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"]
        }

        output_format = config.map_response_format_to_anthropic_output_format(
            response_format
        )

        assert output_format is not None
        transformed_schema = output_format["schema"]

        # minimum/maximum should be removed for Anthropic
        age_schema = transformed_schema["properties"]["age"]
        assert "minimum" not in age_schema
        assert "maximum" not in age_schema
        # But type and description should remain
        assert "description" in age_schema

        score_schema = transformed_schema["properties"]["score"]
        assert "minimum" not in score_schema
        assert "maximum" not in score_schema

    def test_numeric_gt_lt_constraints_filtered(self):
        """
        Test that gt/lt constraints (exclusiveMinimum/exclusiveMaximum) are filtered out.

        Pydantic's gt/lt generate exclusiveMinimum/exclusiveMaximum in JSON schema,
        which Anthropic also doesn't support.

        Related issue: https://github.com/BerriAI/litellm/issues/21016
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class ResponseModel(BaseModel):
            temperature: float = Field(gt=0.0, lt=2.0, description="Temperature")

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(ResponseModel)

        # Verify Pydantic generates exclusiveMinimum/exclusiveMaximum in raw schema
        raw_schema = json_schema["json_schema"]["schema"]
        assert "exclusiveMinimum" in raw_schema["properties"]["temperature"]
        assert "exclusiveMaximum" in raw_schema["properties"]["temperature"]

        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"]
        }

        output_format = config.map_response_format_to_anthropic_output_format(
            response_format
        )

        assert output_format is not None
        transformed_schema = output_format["schema"]

        # exclusiveMinimum/exclusiveMaximum should be removed
        temp_schema = transformed_schema["properties"]["temperature"]
        assert "exclusiveMinimum" not in temp_schema
        assert "exclusiveMaximum" not in temp_schema
        assert "description" in temp_schema
