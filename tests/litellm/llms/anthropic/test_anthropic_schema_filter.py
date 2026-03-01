"""
Tests for Anthropic JSON schema filtering.

Ported from anthropic.lib._parse._transform.transform_schema behavior.
See: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#how-sdk-transformation-works
"""

import pytest
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestFilterAnthropicOutputSchema:
    """Test the filter_anthropic_output_schema whitelist-based transform."""

    def test_removes_numeric_constraints_to_description(self):
        """Numeric constraints go to description (SDK format: {key: value})."""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                    "description": "Person's age",
                },
                "score": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 100,
                },
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        # Constraints removed from schema
        assert "minimum" not in result["properties"]["age"]
        assert "maximum" not in result["properties"]["age"]
        assert "exclusiveMinimum" not in result["properties"]["score"]
        assert "exclusiveMaximum" not in result["properties"]["score"]

        # Type preserved
        assert result["properties"]["age"]["type"] == "integer"

        # Constraints appended to description in SDK format
        age_desc = result["properties"]["age"]["description"]
        assert "Person's age" in age_desc
        assert "minimum: 0" in age_desc
        assert "maximum: 150" in age_desc

        score_desc = result["properties"]["score"]["description"]
        assert "exclusiveMinimum: 0" in score_desc
        assert "exclusiveMaximum: 100" in score_desc

    def test_removes_string_constraints_to_description(self):
        """String length constraints go to description."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 100}
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "minLength" not in result["properties"]["name"]
        assert "maxLength" not in result["properties"]["name"]
        assert result["properties"]["name"]["type"] == "string"
        name_desc = result["properties"]["name"]["description"]
        assert "minLength: 1" in name_desc
        assert "maxLength: 100" in name_desc

    def test_array_min_items_0_or_1_preserved(self):
        """minItems 0 or 1 are supported by Anthropic and preserved."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["minItems"] == 1
        assert result["type"] == "array"
        assert "description" not in result  # No leftover

    def test_array_min_items_gt_1_to_description(self):
        """minItems > 1 is unsupported and goes to description."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 10,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "minItems" not in result or result.get("minItems") is None
        assert "maxItems" not in result
        assert "minItems: 5" in result["description"]
        assert "maxItems: 10" in result["description"]

    def test_handles_nested_schemas(self):
        """Nested schemas are recursively transformed."""
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
                                "maximum": 100,
                            }
                        },
                    },
                    "minItems": 1,
                }
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        # minItems 1 preserved on array
        assert result["properties"]["items"]["minItems"] == 1

        # Nested numeric constraints removed
        nested_props = result["properties"]["items"]["items"]["properties"]
        assert "minimum" not in nested_props["quantity"]
        assert "maximum" not in nested_props["quantity"]
        assert "minimum: 1" in nested_props["quantity"]["description"]

    def test_oneof_converted_to_anyof(self):
        """oneOf is converted to anyOf (SDK behavior)."""
        schema = {
            "oneOf": [
                {"type": "integer", "minimum": 0},
                {"type": "string", "minLength": 1},
            ]
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        # oneOf -> anyOf conversion
        assert "oneOf" not in result
        assert "anyOf" in result
        assert "minimum" not in result["anyOf"][0]
        assert "minLength" not in result["anyOf"][1]

    def test_anyof_preserved(self):
        """anyOf is preserved and variants are recursively filtered."""
        schema = {
            "anyOf": [
                {"type": "integer", "minimum": 0},
                {"type": "string", "minLength": 1},
            ]
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "anyOf" in result
        assert "minimum" not in result["anyOf"][0]
        assert "minLength" not in result["anyOf"][1]

    def test_allof_preserved(self):
        """allOf is preserved and variants are recursively filtered."""
        schema = {
            "allOf": [
                {"type": "object", "properties": {"x": {"type": "integer", "minimum": 0}}},
            ]
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "allOf" in result
        props = result["allOf"][0]["properties"]
        assert "minimum" not in props["x"]

    def test_forces_additional_properties_false(self):
        """additionalProperties is forced to false on all objects."""
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"val": {"type": "string"}},
                    "additionalProperties": True,
                }
            },
            "additionalProperties": True,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["additionalProperties"] is False
        assert result["properties"]["nested"]["additionalProperties"] is False

    def test_preserves_enum_const_default(self):
        """enum, const, default are preserved (litellm deviation from SDK)."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"],
                    "description": "Current status",
                },
                "version": {"type": "string", "const": "v1"},
                "count": {"type": "integer", "default": 0},
            },
            "required": ["status"],
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["properties"]["status"]["enum"] == ["active", "inactive"]
        assert result["properties"]["version"]["const"] == "v1"
        assert result["properties"]["count"]["default"] == 0

    def test_filters_string_formats(self):
        """Only supported string formats are kept; others go to description."""
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
                "ts": {"type": "string", "format": "date-time"},
                "binary_data": {"type": "string", "format": "binary"},
                "custom": {"type": "string", "format": "custom-format"},
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["properties"]["email"]["format"] == "email"
        assert result["properties"]["ts"]["format"] == "date-time"
        assert "format" not in result["properties"]["binary_data"] or \
            result["properties"]["binary_data"].get("format") != "binary"
        assert "format: binary" in result["properties"]["binary_data"]["description"]
        assert "format: custom-format" in result["properties"]["custom"]["description"]

    def test_ref_passthrough(self):
        """$ref schemas are passed through without modification."""
        schema = {"$ref": "#/$defs/MyModel"}

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result == {"$ref": "#/$defs/MyModel"}

    def test_defs_recursively_transformed(self):
        """$defs entries are recursively transformed."""
        schema = {
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "price": {
                            "type": "number",
                            "minimum": 0,
                            "description": "Price in USD",
                        }
                    },
                    "required": ["price"],
                }
            },
            "type": "object",
            "properties": {"item": {"$ref": "#/$defs/Item"}},
            "required": ["item"],
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        item_def = result["$defs"]["Item"]
        assert "minimum" not in item_def["properties"]["price"]
        assert "minimum: 0" in item_def["properties"]["price"]["description"]
        assert item_def["additionalProperties"] is False
        assert result["additionalProperties"] is False

    def test_deeply_nested_objects_get_additional_properties_false(self):
        """All nested objects get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "properties": {
                                "c": {"type": "integer", "minimum": 1}
                            },
                        }
                    },
                }
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["additionalProperties"] is False
        assert result["properties"]["a"]["additionalProperties"] is False
        assert result["properties"]["a"]["properties"]["b"]["additionalProperties"] is False
        assert "minimum" not in result["properties"]["a"]["properties"]["b"]["properties"]["c"]

    def test_preserves_title(self):
        """title field is preserved."""
        schema = {
            "type": "object",
            "title": "MySchema",
            "properties": {"x": {"type": "string", "title": "X Field"}},
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["title"] == "MySchema"
        assert result["properties"]["x"]["title"] == "X Field"

    def test_non_dict_passthrough(self):
        """Non-dict inputs are returned as-is."""
        assert AnthropicConfig.filter_anthropic_output_schema("string") == "string"
        assert AnthropicConfig.filter_anthropic_output_schema(42) == 42
        assert AnthropicConfig.filter_anthropic_output_schema(None) is None

    def test_unsupported_keywords_go_to_description(self):
        """Any unknown/unsupported JSON Schema keywords go to description."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "pattern": "^[a-z]+$",
                    "contentMediaType": "text/plain",
                }
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        data_desc = result["properties"]["data"]["description"]
        assert "pattern: ^[a-z]+$" in data_desc
        assert "contentMediaType: text/plain" in data_desc


class TestMapResponseFormatSchemaFiltering:
    """Test that response_format -> tool conversion filters unsupported schema constraints."""

    def _make_response_format(self, schema: dict) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": schema,
            },
        }

    def test_anthropic_tool_path_filters_constraints(self):
        """map_response_format_to_anthropic_tool strips minimum/maximum etc."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
            },
            "required": ["age", "name"],
        }
        config = AnthropicConfig()
        tool = config.map_response_format_to_anthropic_tool(
            self._make_response_format(schema), {}, False
        )
        assert tool is not None
        input_schema = tool["input_schema"]
        assert "minimum" not in input_schema["properties"]["age"]
        assert "maximum" not in input_schema["properties"]["age"]
        assert "minLength" not in input_schema["properties"]["name"]
        assert "maxLength" not in input_schema["properties"]["name"]
        # Constraints in description
        assert "minimum: 0" in input_schema["properties"]["age"]["description"]

    def test_anthropic_output_format_path_filters_constraints(self):
        """map_response_format_to_anthropic_output_format strips constraints."""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
            },
            "required": ["score"],
        }
        config = AnthropicConfig()
        result = config.map_response_format_to_anthropic_output_format(
            self._make_response_format(schema)
        )
        assert result is not None
        assert "minimum" not in result["schema"]["properties"]["score"]
        assert "maximum" not in result["schema"]["properties"]["score"]

    def test_bedrock_translate_response_format_filters_constraints(self):
        """Bedrock _translate_response_format_param strips constraints for both paths."""
        from litellm.llms.bedrock.chat.converse_transformation import (
            AmazonConverseConfig,
        )

        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 50},
                "items": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": 10,
                },
            },
            "required": ["count", "items"],
        }
        value = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": schema},
        }
        config = AmazonConverseConfig()

        # Tool fallback path (model that doesn't support native structured outputs)
        result = config._translate_response_format_param(
            value=value,
            model="anthropic.claude-3-haiku-20240307-v1:0",
            optional_params={},
            non_default_params={},
            is_thinking_enabled=False,
        )
        # The tool should have been created with filtered schema
        assert "tools" in result
        tool_params = result["tools"][0]["function"]["parameters"]
        assert "minimum" not in tool_params["properties"]["count"]
        assert "maximum" not in tool_params["properties"]["count"]
        # minItems 1 is preserved (supported)
        assert tool_params["properties"]["items"].get("minItems") == 1
        # maxItems is unsupported -> removed
        assert "maxItems" not in tool_params["properties"]["items"]
        nested_items_schema = tool_params["properties"]["items"]["items"]
        assert "minLength" not in nested_items_schema

    def test_strict_tool_filters_nested_constraints(self):
        """_map_tool_helper filters nested constraints when strict: true."""
        tool = {
            "type": "function",
            "function": {
                "name": "get_user",
                "description": "Get user info",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "minimum": 0, "maximum": 200},
                        "name": {"type": "string", "minLength": 1},
                    },
                    "required": ["age", "name"],
                },
            },
        }
        config = AnthropicConfig()
        result_tool, _ = config._map_tool_helper(tool)
        assert result_tool is not None
        schema = result_tool["input_schema"]
        assert "minimum" not in schema["properties"]["age"]
        assert "maximum" not in schema["properties"]["age"]
        assert "minLength" not in schema["properties"]["name"]
        assert "minimum: 0" in schema["properties"]["age"]["description"]

    def test_non_strict_tool_preserves_nested_constraints(self):
        """_map_tool_helper does NOT filter nested constraints without strict: true."""
        tool = {
            "type": "function",
            "function": {
                "name": "get_user",
                "description": "Get user info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "minimum": 0, "maximum": 200},
                    },
                    "required": ["age"],
                },
            },
        }
        config = AnthropicConfig()
        result_tool, _ = config._map_tool_helper(tool)
        assert result_tool is not None
        schema = result_tool["input_schema"]
        # Non-strict tools should keep nested constraints (Anthropic ignores them)
        assert schema["properties"]["age"]["minimum"] == 0
        assert schema["properties"]["age"]["maximum"] == 200
