"""
Tests for LiteLLMCompletionResponsesConfig._clean_schema method.

Tests the schema cleaning logic that removes unsupported fields
(strict, additionalProperties) for domestic models.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


class TestCleanSchema(unittest.TestCase):
    """Test _clean_schema static method."""

    def test_remove_strict_field(self):
        """Test that 'strict' field is removed from schema."""
        schema = {
            "type": "object",
            "strict": True,
            "properties": {"name": {"type": "string"}},
        }
        result = LiteLLMCompletionResponsesConfig._clean_schema(schema)
        self.assertNotIn("strict", result)
        self.assertEqual(result["type"], "object")
        self.assertIn("properties", result)

    def test_remove_additional_properties_field(self):
        """Test that 'additionalProperties' field is removed."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"id": {"type": "integer"}},
        }
        result = LiteLLMCompletionResponsesConfig._clean_schema(schema)
        self.assertNotIn("additionalProperties", result)

    def test_nested_dict_cleaning(self):
        """Test that nested dicts are also cleaned."""
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "strict": True,
                    "additionalProperties": True,
                    "properties": {"value": {"type": "string"}},
                }
            },
        }
        result = LiteLLMCompletionResponsesConfig._clean_schema(schema)
        self.assertNotIn("strict", result["properties"]["nested"])
        self.assertNotIn("additionalProperties", result["properties"]["nested"])

    def test_list_cleaning(self):
        """Test that items in lists are also cleaned."""
        schema = {
            "anyOf": [
                {"type": "string", "strict": True},
                {"type": "integer", "strict": False},
            ]
        }
        result = LiteLLMCompletionResponsesConfig._clean_schema(schema)
        self.assertNotIn("strict", result["anyOf"][0])
        self.assertNotIn("strict", result["anyOf"][1])

    def test_deep_nested_cleaning(self):
        """Test deep nesting is properly cleaned."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "strict": True,
                    "properties": {
                        "level2": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "level3": {
                                    "type": "string",
                                    "strict": True,
                                }
                            },
                        }
                    },
                }
            },
        }
        result = LiteLLMCompletionResponsesConfig._clean_schema(schema)
        # Level 1
        self.assertNotIn("strict", result["properties"]["level1"])
        # Level 2
        level2 = result["properties"]["level1"]["properties"]["level2"]
        self.assertNotIn("additionalProperties", level2)
        # Level 3
        level3 = level2["properties"]["level3"]
        self.assertNotIn("strict", level3)

    def test_non_dict_input(self):
        """Test that non-dict inputs are returned unchanged."""
        self.assertEqual(
            LiteLLMCompletionResponsesConfig._clean_schema("string"), "string"
        )
        self.assertEqual(LiteLLMCompletionResponsesConfig._clean_schema(123), 123)
        self.assertEqual(LiteLLMCompletionResponsesConfig._clean_schema(None), None)
        self.assertEqual(
            LiteLLMCompletionResponsesConfig._clean_schema(["a", "b"]), ["a", "b"]
        )

    def test_empty_dict(self):
        """Test that empty dict returns empty dict."""
        result = LiteLLMCompletionResponsesConfig._clean_schema({})
        self.assertEqual(result, {})

    def test_dict_without_target_fields(self):
        """Test that dict without target fields is unchanged."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = LiteLLMCompletionResponsesConfig._clean_schema(schema)
        self.assertEqual(result, schema)

    def test_preserve_other_fields(self):
        """Test that other fields are preserved."""
        schema = {
            "type": "object",
            "strict": True,
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
            "description": "Test schema",
        }
        result = LiteLLMCompletionResponsesConfig._clean_schema(schema)
        self.assertIn("type", result)
        self.assertIn("properties", result)
        self.assertIn("required", result)
        self.assertIn("description", result)


if __name__ == "__main__":
    unittest.main()
