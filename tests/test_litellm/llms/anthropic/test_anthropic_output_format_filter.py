"""
Coverage for filter_anthropic_output_schema's array/object constraint stripping.

Mirrors tests/litellm/llms/anthropic/test_anthropic_schema_filter.py, but lives
under tests/test_litellm/ so the coverage-uploading CI job exercises the stripped
keyword handling (uniqueItems / contains / minProperties / maxProperties plus
multipleOf / patternProperties / propertyNames / dependentRequired /
dependentSchemas / unevaluatedProperties / if / then / else / not / prefixItems),
the ``uniqueItems: false`` branch, the oneOf to anyOf rewrite, and the
deterministic note ordering.
"""

from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestOutputFormatArrayObjectConstraints:
    def test_removes_uniqueitems(self):
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
        assert "all array items must be unique" in result["properties"]["tags"]["description"]

    def test_uniqueitems_false_skips_misleading_note(self):
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": False,
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        assert "uniqueItems" not in result
        assert "unique" not in result.get("description", "")

    def test_removes_contains_constraints(self):
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
        assert "array must contain an item matching:" in result["description"]
        assert '"const": 1' in result["description"]

    def test_removes_object_property_constraints(self):
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

    def test_removes_remaining_rejected_keywords(self):
        schema = {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "multipleOf": 5},
                "pair": {"type": "array", "prefixItems": [{"type": "number"}], "items": {"type": "number"}},
                "color": {"type": "string", "not": {"const": "red"}},
            },
            "patternProperties": {"^x": {"type": "string"}},
            "propertyNames": {"pattern": "^[a-z]+$"},
            "dependentRequired": {"n": ["pair"]},
            "dependentSchemas": {"n": {"required": ["pair"]}},
            "unevaluatedProperties": {"type": "string"},
            "if": {"properties": {"n": {"const": 5}}},
            "then": {"required": ["pair"]},
            "else": {"required": ["color"]},
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        for field in (
            "patternProperties",
            "propertyNames",
            "dependentRequired",
            "dependentSchemas",
            "unevaluatedProperties",
            "if",
            "then",
            "else",
        ):
            assert field not in result
        assert "multipleOf" not in result["properties"]["n"]
        assert "must be a multiple of 5" in result["properties"]["n"]["description"]
        assert "prefixItems" not in result["properties"]["pair"]
        assert 'leading items must match, in order: [{"type": "number"}]' in result["properties"]["pair"]["description"]
        assert "not" not in result["properties"]["color"]
        assert 'must not match: {"const": "red"}' in result["properties"]["color"]["description"]
        assert 'conditional (if): {"properties": {"n": {"const": 5}}}' in result["description"]

    def test_oneof_rewritten_to_anyof(self):
        schema = {
            "type": "object",
            "properties": {"id": {"oneOf": [{"type": "string", "minLength": 1}, {"type": "integer"}]}},
        }

        result = AnthropicConfig.filter_anthropic_output_schema(schema)

        id_schema = result["properties"]["id"]
        assert "oneOf" not in id_schema
        assert [v["type"] for v in id_schema["anyOf"]] == ["string", "integer"]
        assert "minLength" not in id_schema["anyOf"][0]

    def test_constraint_note_order_is_deterministic(self):
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
