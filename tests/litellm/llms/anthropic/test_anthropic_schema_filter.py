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
            "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 100}},
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
            "maxItems": 10,
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
                                "maximum": 100,
                            }
                        },
                    },
                    "minItems": 1,
                }
            },
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
                {"type": "string", "minLength": 1},
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
                    "description": "Current status",
                }
            },
            "required": ["status"],
            "additionalProperties": False,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result == schema  # Should be unchanged

    def test_removes_uniqueitems(self):
        """Test that uniqueItems is removed from array schemas.

        Reproduces the 400 ``invalid_request_error``:
        "output_format.schema: For 'array' type, property 'uniqueItems' is not
        supported".
        """
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                }
            },
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "uniqueItems" not in result["properties"]["tags"]
        assert result["properties"]["tags"]["items"] == {"type": "string"}
        # Constraint intent preserved in the description
        assert "all array items must be unique" in result["properties"]["tags"]["description"]

    def test_removes_contains_constraints(self):
        """Test that contains/minContains/maxContains are removed from arrays."""
        schema = {
            "type": "array",
            "items": {"type": "integer"},
            "contains": {"type": "integer", "const": 1},
            "minContains": 1,
            "maxContains": 3,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "contains" not in result
        assert "minContains" not in result
        assert "maxContains" not in result
        assert result["items"] == {"type": "integer"}
        # The contains sub-schema is serialized into the advisory note so the model
        # knows what item the array must contain.
        assert "array must contain an item matching:" in result["description"]
        assert '"const": 1' in result["description"]
        assert "minimum number of matching items: 1" in result["description"]
        assert "maximum number of matching items: 3" in result["description"]

    def test_removes_object_property_constraints(self):
        """Test that minProperties/maxProperties are removed from object schemas."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "minProperties": 1,
            "maxProperties": 5,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "minProperties" not in result
        assert "maxProperties" not in result
        assert "minimum number of properties: 1" in result["description"]
        assert "maximum number of properties: 5" in result["description"]

    def test_uniqueitems_false_skips_misleading_note(self):
        """``uniqueItems: false`` is stripped but must not add a 'unique' note."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": False,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "uniqueItems" not in result
        # A disabled constraint imposes no requirement -> no advisory note
        assert "unique" not in result.get("description", "")

    def test_removes_multipleof(self):
        """multipleOf is rejected by Anthropic for integer and number types."""
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer", "multipleOf": 5}},
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "multipleOf" not in result["properties"]["n"]
        assert "must be a multiple of 5" in result["properties"]["n"]["description"]

    def test_removes_conditional_and_negation_keywords(self):
        """if/then/else and not are rejected by Anthropic and stripped into notes."""
        schema = {
            "type": "object",
            "properties": {"kind": {"type": "string"}, "sound": {"type": "string", "not": {"const": "moo"}}},
            "if": {"properties": {"kind": {"const": "dog"}}},
            "then": {"required": ["sound"]},
            "else": {"required": ["kind"]},
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "if" not in result
        assert "then" not in result
        assert "else" not in result
        assert "not" not in result["properties"]["sound"]
        assert 'conditional (if): {"properties": {"kind": {"const": "dog"}}}' in result["description"]
        assert 'conditional (then): {"required": ["sound"]}' in result["description"]
        assert 'conditional (else): {"required": ["kind"]}' in result["description"]
        assert 'must not match: {"const": "moo"}' in result["properties"]["sound"]["description"]

    def test_removes_object_shape_keywords(self):
        """patternProperties/propertyNames/dependent*/unevaluatedProperties are stripped."""
        schema = {
            "type": "object",
            "properties": {"first": {"type": "string"}},
            "patternProperties": {"^x": {"type": "string"}},
            "propertyNames": {"pattern": "^[a-z]+$"},
            "dependentRequired": {"first": ["last"]},
            "dependentSchemas": {"first": {"required": ["last"]}},
            "unevaluatedProperties": {"type": "string"},
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        for field in (
            "patternProperties",
            "propertyNames",
            "dependentRequired",
            "dependentSchemas",
            "unevaluatedProperties",
        ):
            assert field not in result
        assert 'properties whose names match each pattern must satisfy: {"^x": {"type": "string"}}' in result["description"]
        assert 'property names must satisfy: {"pattern": "^[a-z]+$"}' in result["description"]
        assert 'dependent required properties: {"first": ["last"]}' in result["description"]
        assert 'dependent schemas: {"first": {"required": ["last"]}}' in result["description"]
        assert 'unevaluated properties must satisfy: {"type": "string"}' in result["description"]

    def test_removes_prefixitems(self):
        """prefixItems is rejected by Anthropic for array types."""
        schema = {
            "type": "array",
            "prefixItems": [{"type": "number"}, {"type": "string"}],
            "items": {"type": "number"},
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "prefixItems" not in result
        assert result["items"] == {"type": "number"}
        assert 'leading items must match, in order: [{"type": "number"}, {"type": "string"}]' in result["description"]

    def test_oneof_rewritten_to_anyof(self):
        """oneOf 400s ("Schema type 'oneOf' is not supported") and becomes anyOf, like the SDK."""
        schema = {
            "type": "object",
            "properties": {"id": {"oneOf": [{"type": "string", "minLength": 1}, {"type": "integer"}]}},
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        id_schema = result["properties"]["id"]
        assert "oneOf" not in id_schema
        assert [v["type"] for v in id_schema["anyOf"]] == ["string", "integer"]
        assert "minLength" not in id_schema["anyOf"][0]
        assert "minimum length: 1" in id_schema["anyOf"][0]["description"]

    def test_oneof_merges_into_existing_anyof(self):
        schema = {
            "anyOf": [{"type": "string"}],
            "oneOf": [{"type": "integer"}],
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "oneOf" not in result
        assert [v["type"] for v in result["anyOf"]] == ["string", "integer"]

    def test_constraint_note_order_is_deterministic(self):
        """Note order must not depend on set iteration order (PYTHONHASHSEED), or the
        serialized request differs across proxy workers and breaks caching."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
            "uniqueItems": True,
            "minContains": 2,
            "maxContains": 3,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert result["description"] == (
            "Note: minimum number of items: 1, maximum number of items: 10, "
            "all array items must be unique, minimum number of matching items: 2, "
            "maximum number of matching items: 3."
        )
