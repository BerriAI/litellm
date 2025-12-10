"""
Test nested path support in additional_drop_params.

This tests the new JSONPath-like syntax for removing nested fields.
"""

import os
import sys


# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from litellm.litellm_core_utils.dot_notation_indexing import (
    delete_nested_value,
    is_nested_path,
)


class TestIsNestedPath:
    """Test path detection."""

    def test_top_level_path(self):
        """Top-level paths should return False."""
        assert is_nested_path("temperature") is False
        assert is_nested_path("response_format") is False

    def test_nested_path_with_dot(self):
        """Paths with dots are nested."""
        assert is_nested_path("parent.child") is True

    def test_nested_path_with_array(self):
        """Paths with array notation are nested."""
        assert is_nested_path("tools[*].input_examples") is True
        assert is_nested_path("tools[0].field") is True


class TestDeleteNestedValue:
    """Test the core deletion logic."""

    def test_array_wildcard_removes_field_from_all_elements(self):
        """Test removing a field from all array elements."""
        data = {
            "tools": [
                {"name": "tool1", "input_examples": ["ex1"]},
                {"name": "tool2", "input_examples": ["ex2"]},
            ],
            "temperature": 0.7,
        }

        result = delete_nested_value(data, "tools[*].input_examples")

        # Verify structure preserved
        assert len(result["tools"]) == 2
        assert result["tools"][0]["name"] == "tool1"
        assert result["tools"][1]["name"] == "tool2"
        assert result["temperature"] == 0.7

        # Verify input_examples removed
        assert "input_examples" not in result["tools"][0]
        assert "input_examples" not in result["tools"][1]

        # Verify original unchanged (deep copy)
        assert "input_examples" in data["tools"][0]

    def test_specific_array_index_removes_field_from_single_element(self):
        """Test removing a field from specific array elements using [n] syntax."""
        # Test data with multiple array elements
        data = {
            "tools": [
                {"name": "t0", "input_examples": ["ex0"], "keep": "val0"},
                {"name": "t1", "input_examples": ["ex1"], "keep": "val1"},
                {"name": "t2", "input_examples": ["ex2"], "keep": "val2"},
                {"name": "t3", "input_examples": ["ex3"], "keep": "val3"},
                {"name": "t4", "input_examples": ["ex4"], "keep": "val4"},
                {"name": "t5", "input_examples": ["ex5"], "keep": "val5"},
            ]
        }

        # Test [0] - first element
        result = delete_nested_value(data, "tools[0].input_examples")
        assert "input_examples" not in result["tools"][0]
        assert "input_examples" in result["tools"][1]
        assert "input_examples" in result["tools"][2]

        # Test [1] - second element
        result = delete_nested_value(data, "tools[1].input_examples")
        assert "input_examples" in result["tools"][0]
        assert "input_examples" not in result["tools"][1]
        assert "input_examples" in result["tools"][2]

        # Test [2] - middle element
        result = delete_nested_value(data, "tools[2].input_examples")
        assert "input_examples" in result["tools"][0]
        assert "input_examples" not in result["tools"][2]
        assert "input_examples" in result["tools"][3]

        # Test [5] - last element
        result = delete_nested_value(data, "tools[5].input_examples")
        assert "input_examples" in result["tools"][0]
        assert "input_examples" not in result["tools"][5]

        # Verify other fields preserved in all cases
        assert result["tools"][0]["keep"] == "val0"
        assert result["tools"][5]["keep"] == "val5"

        # Verify original unchanged (deep copy)
        assert "input_examples" in data["tools"][0]


class TestComplexNestedPatterns:
    """Test complex nested patterns with multiple wildcards and deep nesting."""

    def test_multiple_jsonpath_patterns_in_list(self):
        """Test processing multiple JSONPath patterns sequentially."""
        data = {
            "tools": [
                {
                    "name": "tool1",
                    "input_examples": ["ex1"],
                    "some_arr": [
                        {
                            "some_struct": {
                                "remove_this_field": "val1",
                                "keep_this": "val2",
                            }
                        },
                        {
                            "some_struct": {
                                "remove_this_field": "val3",
                                "keep_this": "val4",
                            }
                        },
                    ],
                },
                {
                    "name": "tool2",
                    "input_examples": ["ex2"],
                    "some_arr": [
                        {
                            "some_struct": {
                                "remove_this_field": "val5",
                                "keep_this": "val6",
                            }
                        }
                    ],
                },
            ],
            "temperature": 0.7,
        }

        # Simulate multiple paths being processed (as in utils.py:4134-4137)
        paths = [
            "tools[*].input_examples",
            "tools[*].some_arr[*].some_struct.remove_this_field",
        ]

        result = data
        for path in paths:
            result = delete_nested_value(result, path)

        # Verify input_examples removed from all tools
        assert "input_examples" not in result["tools"][0]
        assert "input_examples" not in result["tools"][1]

        # Verify deeply nested field removed from all array elements
        assert (
            "remove_this_field"
            not in result["tools"][0]["some_arr"][0]["some_struct"]
        )
        assert (
            "remove_this_field"
            not in result["tools"][0]["some_arr"][1]["some_struct"]
        )
        assert (
            "remove_this_field"
            not in result["tools"][1]["some_arr"][0]["some_struct"]
        )

        # Verify other fields preserved
        assert result["tools"][0]["some_arr"][0]["some_struct"]["keep_this"] == "val2"
        assert result["tools"][1]["some_arr"][0]["some_struct"]["keep_this"] == "val6"
        assert result["temperature"] == 0.7

    def test_remove_entire_nested_array_field(self):
        """Test removing entire array fields (not just array elements)."""
        data = {
            "tools": [
                {"name": "t1", "some_arr": [1, 2, 3], "other_field": "keep"},
                {"name": "t2", "some_arr": [4, 5, 6], "other_field": "keep"},
            ]
        }

        result = delete_nested_value(data, "tools[*].some_arr")

        # Verify entire array field removed (not individual elements)
        assert "some_arr" not in result["tools"][0]
        assert "some_arr" not in result["tools"][1]

        # Verify other fields preserved
        assert result["tools"][0]["name"] == "t1"
        assert result["tools"][0]["other_field"] == "keep"
        assert result["tools"][1]["name"] == "t2"
        assert result["tools"][1]["other_field"] == "keep"

    def test_triple_nested_wildcards(self):
        """Test extreme nesting: tools[*].arr1[*].arr2[*].field."""
        data = {
            "tools": [
                {
                    "name": "t1",
                    "arr1": [
                        {
                            "arr2": [
                                {"field": "remove1", "keep": "yes1"},
                                {"field": "remove2", "keep": "yes2"},
                            ]
                        },
                        {
                            "arr2": [
                                {"field": "remove3", "keep": "yes3"},
                            ]
                        },
                    ],
                }
            ]
        }

        result = delete_nested_value(data, "tools[*].arr1[*].arr2[*].field")

        # Verify deeply nested field removed from all levels
        assert "field" not in result["tools"][0]["arr1"][0]["arr2"][0]
        assert "field" not in result["tools"][0]["arr1"][0]["arr2"][1]
        assert "field" not in result["tools"][0]["arr1"][1]["arr2"][0]

        # Verify keep field preserved at all levels
        assert result["tools"][0]["arr1"][0]["arr2"][0]["keep"] == "yes1"
        assert result["tools"][0]["arr1"][0]["arr2"][1]["keep"] == "yes2"
        assert result["tools"][0]["arr1"][1]["arr2"][0]["keep"] == "yes3"

    def test_combination_of_simple_and_complex_paths(self):
        """Test mixing simple nested paths with complex multi-wildcard paths."""
        data = {
            "tools": [
                {
                    "name": "t1",
                    "simple_nested": {"remove": "val1", "keep": "val2"},
                    "complex": [{"nested": {"remove": "val3", "keep": "val4"}}],
                }
            ],
            "top_level_remove": "should_go",
            "top_level_keep": "should_stay",
        }

        # Process multiple different types of paths
        paths = [
            "tools[*].simple_nested.remove",
            "tools[*].complex[*].nested.remove",
        ]

        result = data
        for path in paths:
            result = delete_nested_value(result, path)

        # Verify simple nested removal
        assert "remove" not in result["tools"][0]["simple_nested"]
        assert result["tools"][0]["simple_nested"]["keep"] == "val2"

        # Verify complex nested removal
        assert "remove" not in result["tools"][0]["complex"][0]["nested"]
        assert result["tools"][0]["complex"][0]["nested"]["keep"] == "val4"

        # Verify top-level fields unchanged
        assert result["top_level_remove"] == "should_go"
        assert result["top_level_keep"] == "should_stay"

    def test_mixed_wildcards_and_indices_with_deep_nesting(self):
        """Test combining [*] wildcards, [n] indices, and deep nesting in complex patterns."""
        data = {
            "tools": [
                {
                    "name": "t0",
                    "configs": [
                        {"id": "c0", "remove_me": "val1", "keep": "yes1"},
                        {"id": "c1", "remove_me": "val2", "keep": "yes2"},
                    ],
                    "metadata": {"drop_this": "meta1", "preserve": "preserve1"},
                },
                {
                    "name": "t1",
                    "configs": [
                        {"id": "c0", "remove_me": "val3", "keep": "yes3"},
                        {"id": "c1", "remove_me": "val4", "keep": "yes4"},
                    ],
                    "metadata": {"drop_this": "meta2", "preserve": "preserve2"},
                },
                {
                    "name": "t2",
                    "configs": [
                        {"id": "c0", "remove_me": "val5", "keep": "yes5"},
                    ],
                    "metadata": {"drop_this": "meta3", "preserve": "preserve3"},
                },
            ]
        }

        # Simulate processing multiple complex paths
        paths = [
            "tools[*].configs[1].remove_me",  # Wildcard + specific index [1] + nested
            "tools[1].metadata.drop_this",  # Specific index + nested
            "tools[*].configs[*].id",  # Double wildcard + nested
        ]

        result = data
        for path in paths:
            result = delete_nested_value(result, path)

        # Verify: tools[*].configs[1].remove_me removed from second config of all tools (that have one)
        assert "remove_me" in result["tools"][0]["configs"][0]  # First config untouched
        assert (
            "remove_me" not in result["tools"][0]["configs"][1]
        )  # Second config removed
        assert "remove_me" in result["tools"][1]["configs"][0]  # First config untouched
        assert (
            "remove_me" not in result["tools"][1]["configs"][1]
        )  # Second config removed
        assert (
            "remove_me" in result["tools"][2]["configs"][0]
        )  # Only has [0], unaffected

        # Verify: tools[1].metadata.drop_this removed only from second tool
        assert "drop_this" in result["tools"][0]["metadata"]
        assert "drop_this" not in result["tools"][1]["metadata"]
        assert "drop_this" in result["tools"][2]["metadata"]

        # Verify: tools[*].configs[*].id removed from all configs in all tools
        assert "id" not in result["tools"][0]["configs"][0]
        assert "id" not in result["tools"][0]["configs"][1]
        assert "id" not in result["tools"][1]["configs"][0]
        assert "id" not in result["tools"][1]["configs"][1]
        assert "id" not in result["tools"][2]["configs"][0]

        # Verify: other fields preserved
        assert result["tools"][0]["configs"][0]["keep"] == "yes1"
        assert result["tools"][1]["configs"][1]["keep"] == "yes4"
        assert result["tools"][0]["metadata"]["preserve"] == "preserve1"
        assert result["tools"][1]["metadata"]["preserve"] == "preserve2"
        assert result["tools"][2]["name"] == "t2"

        # Verify original unchanged
        assert "remove_me" in data["tools"][0]["configs"][1]


# Phase 1 tests - validates core functionality and complex patterns
