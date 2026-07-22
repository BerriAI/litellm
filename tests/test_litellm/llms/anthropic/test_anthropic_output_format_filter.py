"""
Coverage for filter_anthropic_output_schema's array/object constraint stripping.

Mirrors tests/litellm/llms/anthropic/test_anthropic_schema_filter.py, but lives
under tests/test_litellm/ so the coverage-uploading CI job exercises the newly
added keyword handling (uniqueItems / contains / minProperties / maxProperties)
and the ``uniqueItems: false`` branch.
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
