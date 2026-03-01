"""
Tests for Anthropic JSON schema filtering.

Related to: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#how-sdk-transformation-works
"""

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


class TestToolBasedResponseFormatFiltering:
    """Test that tool-based response_format path also filters unsupported constraints.

    When using older Anthropic models that don't support native output_format,
    LiteLLM falls back to tool-based JSON mode. The tool's input_schema must
    also have unsupported constraints stripped.
    """

    def test_tool_response_format_filters_minimum_maximum(self):
        """Tool-based response_format should strip minimum/maximum from nested schemas."""
        config = AnthropicConfig()
        json_schema = {
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
                    "minimum": 0.0,
                    "maximum": 100.0,
                },
            },
            "required": ["age", "score"],
        }

        tool = config._create_json_tool_call_for_response_format(
            json_schema=json_schema
        )

        input_schema = tool["input_schema"]
        assert "minimum" not in input_schema["properties"]["age"]
        assert "maximum" not in input_schema["properties"]["age"]
        assert "minimum" not in input_schema["properties"]["score"]
        assert "maximum" not in input_schema["properties"]["score"]
        # Descriptions should contain constraint info
        assert "minimum value: 0" in input_schema["properties"]["age"]["description"]
        assert "maximum value: 150" in input_schema["properties"]["age"]["description"]

    def test_tool_response_format_filters_string_constraints(self):
        """Tool-based response_format should strip minLength/maxLength."""
        config = AnthropicConfig()
        json_schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 255,
                }
            },
            "required": ["name"],
        }

        tool = config._create_json_tool_call_for_response_format(
            json_schema=json_schema
        )

        input_schema = tool["input_schema"]
        assert "minLength" not in input_schema["properties"]["name"]
        assert "maxLength" not in input_schema["properties"]["name"]

    def test_tool_response_format_filters_array_constraints(self):
        """Tool-based response_format should strip minItems/maxItems."""
        config = AnthropicConfig()
        json_schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
            "required": ["tags"],
        }

        tool = config._create_json_tool_call_for_response_format(
            json_schema=json_schema
        )

        input_schema = tool["input_schema"]
        assert "minItems" not in input_schema["properties"]["tags"]
        assert "maxItems" not in input_schema["properties"]["tags"]

    def test_tool_response_format_preserves_valid_fields(self):
        """Tool-based response_format should preserve valid schema fields."""
        config = AnthropicConfig()
        json_schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"],
                    "description": "Current status",
                }
            },
            "required": ["status"],
        }

        tool = config._create_json_tool_call_for_response_format(
            json_schema=json_schema
        )

        input_schema = tool["input_schema"]
        assert input_schema["properties"]["status"]["type"] == "string"
        assert input_schema["properties"]["status"]["enum"] == ["active", "inactive"]
        assert input_schema["properties"]["status"]["description"] == "Current status"


class TestRegularToolSchemaFiltering:
    """Test that regular tool schemas also filter unsupported constraints.

    When tools are mapped via _map_tool_helper, nested schemas within
    properties should have unsupported constraints stripped.
    """

    def _make_openai_tool(self, name, parameters):
        """Helper to create an OpenAI-format tool definition."""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Test tool {name}",
                "parameters": parameters,
            },
        }

    def test_tool_filters_nested_numeric_constraints(self):
        """Regular tools should strip minimum/maximum from nested property schemas."""
        config = AnthropicConfig()
        tool = self._make_openai_tool(
            "create_user",
            {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 150,
                        "description": "User age",
                    }
                },
                "required": ["age"],
            },
        )

        result, _ = config._map_tool_helper(tool)
        assert result is not None
        props = result["input_schema"]["properties"]
        assert "minimum" not in props["age"]
        assert "maximum" not in props["age"]
        assert "User age" in props["age"]["description"]
        assert "minimum value: 0" in props["age"]["description"]

    def test_tool_filters_nested_string_constraints(self):
        """Regular tools should strip minLength/maxLength from nested property schemas."""
        config = AnthropicConfig()
        tool = self._make_openai_tool(
            "search",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    }
                },
                "required": ["query"],
            },
        )

        result, _ = config._map_tool_helper(tool)
        assert result is not None
        props = result["input_schema"]["properties"]
        assert "minLength" not in props["query"]
        assert "maxLength" not in props["query"]

    def test_tool_filters_deeply_nested_constraints(self):
        """Regular tools should strip constraints from deeply nested schemas."""
        config = AnthropicConfig()
        tool = self._make_openai_tool(
            "create_order",
            {
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
                                    "maximum": 1000,
                                }
                            },
                        },
                        "minItems": 1,
                        "maxItems": 50,
                    }
                },
                "required": ["items"],
            },
        )

        result, _ = config._map_tool_helper(tool)
        assert result is not None
        items_schema = result["input_schema"]["properties"]["items"]
        assert "minItems" not in items_schema
        assert "maxItems" not in items_schema
        nested_props = items_schema["items"]["properties"]
        assert "minimum" not in nested_props["quantity"]
        assert "maximum" not in nested_props["quantity"]

    def test_tool_preserves_valid_nested_fields(self):
        """Regular tools should preserve valid schema fields in nested schemas."""
        config = AnthropicConfig()
        tool = self._make_openai_tool(
            "set_status",
            {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Device status",
                    }
                },
                "required": ["status"],
            },
        )

        result, _ = config._map_tool_helper(tool)
        assert result is not None
        props = result["input_schema"]["properties"]
        assert props["status"]["type"] == "string"
        assert props["status"]["enum"] == ["on", "off"]
        assert props["status"]["description"] == "Device status"


class TestAdditionalSDKTransformations:
    """Test additional SDK transformations: additionalProperties, string formats, pattern.

    Mirrors the Anthropic Python SDK's transform_schema() behavior:
    - Add additionalProperties: false to all object schemas
    - Filter string format to supported values only
    - Strip pattern constraint (move to description)
    See: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/_parse/_transform.py
    """

    def test_adds_additional_properties_false_to_objects(self):
        """Object schemas should get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"],
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["additionalProperties"] is False

    def test_overrides_additional_properties_true(self):
        """additionalProperties: true should be overridden to false for objects."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": True,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["additionalProperties"] is False

    def test_preserves_additional_properties_false(self):
        """additionalProperties: false should remain false."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": False,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["additionalProperties"] is False

    def test_no_additional_properties_for_non_object(self):
        """Non-object schemas should NOT get additionalProperties."""
        schema = {"type": "string"}

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "additionalProperties" not in result

    def test_additional_properties_on_nested_objects(self):
        """Nested object schemas should also get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                }
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["additionalProperties"] is False
        assert result["properties"]["address"]["additionalProperties"] is False

    def test_preserves_supported_string_format(self):
        """Supported string formats (date-time, email, uuid, etc.) should be preserved."""
        for fmt in ["date-time", "date", "time", "email", "uri", "uuid", "ipv4", "ipv6", "hostname", "duration"]:
            schema = {"type": "string", "format": fmt}
            result = AnthropicConfig.filter_anthropic_output_schema(schema)
            assert result.get("format") == fmt, f"Format {fmt} should be preserved"

    def test_strips_unsupported_string_format(self):
        """Unsupported string formats should be removed and added to description."""
        schema = {
            "type": "string",
            "format": "phone-number",
            "description": "Contact phone",
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "format" not in result
        assert "phone-number" in result["description"]
        assert "Contact phone" in result["description"]

    def test_strips_unsupported_format_no_existing_description(self):
        """Unsupported format stripped without existing description."""
        schema = {"type": "string", "format": "binary"}

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "format" not in result
        assert "binary" in result["description"]

    def test_format_preserved_for_non_string_types(self):
        """Format on non-string types should pass through unchanged."""
        schema = {"type": "integer", "format": "int64"}

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result.get("format") == "int64"

    def test_strips_pattern_constraint(self):
        """pattern constraint should be stripped and moved to description."""
        schema = {
            "type": "string",
            "pattern": "^[A-Z]{2}\\d{4}$",
            "description": "Product code",
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "pattern" not in result
        assert "Product code" in result["description"]
        assert "pattern:" in result["description"]

    def test_combined_transformations(self):
        """Test all transformations together on a complex schema."""
        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                    "minLength": 5,
                },
                "phone": {
                    "type": "string",
                    "format": "phone-number",
                    "pattern": "^\\+\\d+$",
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 50},
                    "minItems": 1,
                    "maxItems": 10,
                },
            },
            "required": ["email", "phone", "age", "tags"],
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        # Object gets additionalProperties: false
        assert result["additionalProperties"] is False

        # email: supported format preserved, minLength stripped
        assert result["properties"]["email"]["format"] == "email"
        assert "minLength" not in result["properties"]["email"]
        assert "minimum length: 5" in result["properties"]["email"]["description"]

        # phone: unsupported format and pattern stripped
        assert "format" not in result["properties"]["phone"]
        assert "pattern" not in result["properties"]["phone"]
        assert "phone-number" in result["properties"]["phone"]["description"]

        # age: numeric constraints stripped
        assert "minimum" not in result["properties"]["age"]
        assert "maximum" not in result["properties"]["age"]

        # tags: array constraints stripped, nested string constraint stripped
        assert "minItems" not in result["properties"]["tags"]
        assert "maxItems" not in result["properties"]["tags"]
        assert "maxLength" not in result["properties"]["tags"]["items"]


class TestTupleAndPrefixItemsFiltering:
    """Tests for tuple-style items (list) and prefixItems filtering."""

    def test_tuple_style_items_list_filtered(self):
        """Tuple-style items (list of schemas) should be recursively filtered."""
        schema = {
            "type": "array",
            "items": [
                {"type": "number", "minimum": -90, "maximum": 90},
                {"type": "number", "minimum": -180, "maximum": 180},
            ],
        }
        result = AnthropicConfig.filter_anthropic_output_schema(schema)
        for item in result["items"]:
            assert "minimum" not in item
            assert "maximum" not in item
            assert "description" in item

    def test_prefix_items_filtered(self):
        """prefixItems (JSON Schema 2020-12) should be recursively filtered."""
        schema = {
            "type": "array",
            "prefixItems": [
                {"type": "string", "minLength": 1, "maxLength": 10},
                {"type": "integer", "minimum": 0},
            ],
        }
        result = AnthropicConfig.filter_anthropic_output_schema(schema)
        for item in result["prefixItems"]:
            assert "minLength" not in item
            assert "maxLength" not in item
            assert "minimum" not in item


class TestEnforceAdditionalPropertiesFlag:
    """Test the enforce_additional_properties parameter."""

    def test_enforce_false_preserves_additional_properties_true(self):
        """When enforce_additional_properties=False, user's additionalProperties: true is preserved."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": True,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(
            schema, enforce_additional_properties=False
        )

        assert result["additionalProperties"] is True

    def test_enforce_false_does_not_add_additional_properties(self):
        """When enforce_additional_properties=False, additionalProperties is NOT injected."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(
            schema, enforce_additional_properties=False
        )

        assert "additionalProperties" not in result

    def test_enforce_false_still_filters_constraints(self):
        """Constraint filtering still works when enforce_additional_properties=False."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(
            schema, enforce_additional_properties=False
        )

        assert "minimum" not in result["properties"]["age"]
        assert "maximum" not in result["properties"]["age"]
        assert "additionalProperties" not in result

    def test_enforce_false_nested_objects(self):
        """Nested objects also respect enforce_additional_properties=False."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "additionalProperties": True,
                }
            },
            "additionalProperties": True,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(
            schema, enforce_additional_properties=False
        )

        assert result["additionalProperties"] is True
        assert result["properties"]["address"]["additionalProperties"] is True

    def test_enforce_true_is_default(self):
        """Default behavior (enforce_additional_properties=True) forces false."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": True,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["additionalProperties"] is False
