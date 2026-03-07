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

    def test_other_constraints_preserved(self):
        """
        Test that constraints are properly handled (removed from schema, added to description).

        Per Anthropic API requirements, constraints like minLength/maxLength and
        minimum/maximum must be removed from the schema but documented in descriptions.
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

        # String constraints should be REMOVED from schema (Anthropic doesn't support them)
        name_schema = transformed_schema["properties"]["name"]
        assert "maxLength" not in name_schema
        assert "minLength" not in name_schema
        # But constraint info should be added to description
        assert "description" in name_schema
        assert "minimum length: 1" in name_schema["description"]
        assert "maximum length: 100" in name_schema["description"]

        # Number constraints should be REMOVED from schema (Anthropic doesn't support them)
        age_schema = transformed_schema["properties"]["age"]
        assert "minimum" not in age_schema
        assert "maximum" not in age_schema
        # But constraint info should be added to description
        assert "description" in age_schema
        assert "minimum value: 0" in age_schema["description"]
        assert "maximum value: 150" in age_schema["description"]


class TestAnthropicToolBasedStructuredOutput:
    """
    Test that structured output via the tool-based path (older models) also
    filters unsupported JSON schema constraints.

    The output_format path (newer models like Sonnet 4.5+) already filters
    constraints via filter_anthropic_output_schema(). The tool-based path
    for older models was missing this filtering, causing Anthropic API errors
    like: "For 'integer' type, properties maximum, minimum are not supported"

    Related issue: https://github.com/BerriAI/litellm/issues/21016
    """

    def test_numeric_constraints_filtered_in_tool_path(self):
        """
        Test that ge/le/gt/lt constraints on numeric fields are stripped
        when using the tool-based response_format path (older models).
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class Rating(BaseModel):
            score: int = Field(ge=1, le=10, description="Rating score")
            confidence: float = Field(gt=0.0, lt=1.0, description="Confidence")

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(Rating)

        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"],
        }

        tool = config.map_response_format_to_anthropic_tool(
            value=response_format, optional_params={}, is_thinking_enabled=False
        )

        assert tool is not None
        tool_schema = tool["input_schema"]

        # Numeric constraints should be stripped
        score_schema = tool_schema["properties"]["score"]
        assert "minimum" not in score_schema
        assert "maximum" not in score_schema

        confidence_schema = tool_schema["properties"]["confidence"]
        assert "exclusiveMinimum" not in confidence_schema
        assert "exclusiveMaximum" not in confidence_schema

    def test_numeric_constraints_moved_to_description_in_tool_path(self):
        """
        Test that stripped numeric constraints are added to the description
        in the tool-based path, matching the output_format path behavior.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class Rating(BaseModel):
            score: int = Field(ge=1, le=10, description="Rating score")

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(Rating)

        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"],
        }

        tool = config.map_response_format_to_anthropic_tool(
            value=response_format, optional_params={}, is_thinking_enabled=False
        )

        assert tool is not None
        score_schema = tool["input_schema"]["properties"]["score"]
        assert "minimum value: 1" in score_schema["description"]
        assert "maximum value: 10" in score_schema["description"]

    def test_string_constraints_filtered_in_tool_path(self):
        """
        Test that minLength/maxLength constraints on string fields are
        stripped in the tool-based path.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class UserInput(BaseModel):
            name: str = Field(min_length=1, max_length=100, description="User name")

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(UserInput)

        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"],
        }

        tool = config.map_response_format_to_anthropic_tool(
            value=response_format, optional_params={}, is_thinking_enabled=False
        )

        assert tool is not None
        name_schema = tool["input_schema"]["properties"]["name"]
        assert "minLength" not in name_schema
        assert "maxLength" not in name_schema
        assert "minimum length: 1" in name_schema["description"]
        assert "maximum length: 100" in name_schema["description"]

    def test_array_constraints_filtered_in_tool_path(self):
        """
        Test that minItems/maxItems constraints on list fields are
        stripped in the tool-based path.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        class ItemList(BaseModel):
            items: List[str] = Field(min_length=1, max_length=10, description="Items")

        config = AnthropicConfig()
        json_schema = config.get_json_schema_from_pydantic_object(ItemList)

        response_format = {
            "type": "json_schema",
            "json_schema": json_schema["json_schema"],
        }

        tool = config.map_response_format_to_anthropic_tool(
            value=response_format, optional_params={}, is_thinking_enabled=False
        )

        assert tool is not None
        items_schema = tool["input_schema"]["properties"]["items"]
        assert "minItems" not in items_schema
        assert "maxItems" not in items_schema
