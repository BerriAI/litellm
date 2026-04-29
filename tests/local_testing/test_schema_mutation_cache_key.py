"""
Tests for https://github.com/BerriAI/litellm/issues/18784

_remove_strict_from_schema and _remove_additional_properties must NOT mutate
their input, because the caller's original tools dict is used to compute cache
keys both before and after the LLM provider call.  If the schema is mutated
in-place, the cache key at GET time differs from the key at SET time and the
disk cache is effectively broken.
"""

import copy
import tempfile

import pytest

import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.utils import (
    _remove_additional_properties,
    _remove_additional_properties_in_place,
    _remove_strict_from_schema,
    _remove_strict_from_schema_in_place,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tools_with_strict():
    """Return a realistic tools list with strict=True and additionalProperties."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]


# ---------------------------------------------------------------------------
# Core: in-place mutation must not happen
# ---------------------------------------------------------------------------

class TestRemoveStrictDoesNotMutate:
    """_remove_strict_from_schema must return a new object, leaving the input untouched."""

    def test_original_unchanged_after_remove_strict(self):
        tools = _make_tools_with_strict()
        original = copy.deepcopy(tools)

        _remove_strict_from_schema(tools)

        assert tools == original, (
            "_remove_strict_from_schema mutated the input in-place — "
            "this breaks cache-key computation (issue #18784)"
        )

    def test_return_value_has_strict_removed(self):
        tools = _make_tools_with_strict()

        result = _remove_strict_from_schema(tools)

        # 'strict' should be gone from the returned copy
        assert "strict" not in result[0]["function"]

    def test_original_still_has_strict(self):
        tools = _make_tools_with_strict()

        _remove_strict_from_schema(tools)

        # The original must still have 'strict'
        assert tools[0]["function"]["strict"] is True


class TestRemoveAdditionalPropertiesDoesNotMutate:
    """_remove_additional_properties must return a new object, leaving the input untouched."""

    def test_original_unchanged_after_remove_additional_properties(self):
        tools = _make_tools_with_strict()
        original = copy.deepcopy(tools)

        _remove_additional_properties(tools)

        assert tools == original, (
            "_remove_additional_properties mutated the input in-place — "
            "this breaks cache-key computation (issue #18784)"
        )

    def test_return_value_has_additional_properties_removed(self):
        tools = _make_tools_with_strict()

        result = _remove_additional_properties(tools)

        assert "additionalProperties" not in result[0]["function"]["parameters"]

    def test_original_still_has_additional_properties(self):
        tools = _make_tools_with_strict()

        _remove_additional_properties(tools)

        assert tools[0]["function"]["parameters"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Integration: cache key must be stable across remove_strict calls
# ---------------------------------------------------------------------------

class TestCacheKeyStability:
    """The cache key must be identical before and after _remove_strict_from_schema
    is applied to the tools, because the cache GET uses the original tools and
    the cache SET uses whatever is in kwargs after the provider call."""

    def test_cache_key_matches_before_and_after_remove_strict(self):
        tools = _make_tools_with_strict()

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=cache_dir)

            key_before = cache.get_cache_key(
                model="hosted_vllm/test",
                messages=[{"role": "user", "content": "test"}],
                tools=tools,
            )

            # Simulate what the provider does internally
            _remove_strict_from_schema(tools)

            key_after = cache.get_cache_key(
                model="hosted_vllm/test",
                messages=[{"role": "user", "content": "test"}],
                tools=tools,
            )

            assert key_before == key_after, (
                f"Cache key changed after _remove_strict_from_schema: "
                f"{key_before!r} != {key_after!r} — disk cache is broken (issue #18784)"
            )

    def test_cache_key_matches_before_and_after_remove_additional_properties(self):
        tools = _make_tools_with_strict()

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=cache_dir)

            key_before = cache.get_cache_key(
                model="hosted_vllm/test",
                messages=[{"role": "user", "content": "test"}],
                tools=tools,
            )

            # Simulate what the provider does internally
            _remove_additional_properties(tools)

            key_after = cache.get_cache_key(
                model="hosted_vllm/test",
                messages=[{"role": "user", "content": "test"}],
                tools=tools,
            )

            assert key_before == key_after, (
                f"Cache key changed after _remove_additional_properties: "
                f"{key_before!r} != {key_after!r} — disk cache is broken (issue #18784)"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for the deep-copy fix."""

    def test_nested_strict_in_parameters(self):
        """strict can appear at multiple nesting levels."""
        schema = {
            "strict": True,
            "properties": {
                "inner": {
                    "strict": True,
                    "type": "object",
                }
            },
        }
        original = copy.deepcopy(schema)

        result = _remove_strict_from_schema(schema)

        # Original preserved
        assert schema == original
        # Result cleaned
        assert "strict" not in result
        assert "strict" not in result["properties"]["inner"]

    def test_empty_schema(self):
        assert _remove_strict_from_schema({}) == {}
        assert _remove_additional_properties({}) == {}

    def test_none_passthrough(self):
        """Non-dict/list inputs are returned as-is."""
        assert _remove_strict_from_schema(None) is None
        assert _remove_strict_from_schema("hello") == "hello"
        assert _remove_strict_from_schema(42) == 42

    def test_list_of_tools(self):
        """The top-level input is often a list of tool dicts."""
        tools = _make_tools_with_strict()
        original = copy.deepcopy(tools)

        result = _remove_strict_from_schema(tools)

        assert tools == original
        assert "strict" not in result[0]["function"]

    def test_deeply_nested_additional_properties(self):
        """additionalProperties can appear at multiple nesting levels."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outer": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "inner": {
                            "type": "object",
                            "additionalProperties": False,
                        }
                    },
                }
            },
        }
        original = copy.deepcopy(schema)

        result = _remove_additional_properties(schema)

        # Original preserved at all levels
        assert schema == original
        # Result cleaned at all levels
        assert "additionalProperties" not in result
        assert "additionalProperties" not in result["properties"]["outer"]
        assert "additionalProperties" not in result["properties"]["outer"]["properties"]["inner"]


# ---------------------------------------------------------------------------
# Sequential calls: production pattern (hosted_vllm, watsonx call both)
# ---------------------------------------------------------------------------

class TestSequentialCalls:
    """In production, providers like hosted_vllm and watsonx call both
    _remove_additional_properties and _remove_strict_from_schema in sequence
    on the same tools list. The original must survive both calls."""

    def test_original_preserved_after_both_removals(self):
        tools = _make_tools_with_strict()
        original = copy.deepcopy(tools)

        cleaned = _remove_additional_properties(tools)
        cleaned = _remove_strict_from_schema(cleaned)

        assert tools == original, "Original tools mutated by sequential removal calls"
        assert "additionalProperties" not in cleaned[0]["function"]["parameters"]
        assert "strict" not in cleaned[0]["function"]

    def test_cache_key_stable_after_sequential_calls(self):
        """Cache key must be identical before and after both functions run."""
        tools = _make_tools_with_strict()

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=cache_dir)

            key_before = cache.get_cache_key(
                model="hosted_vllm/test",
                messages=[{"role": "user", "content": "test"}],
                tools=tools,
            )

            # Simulate the exact production call sequence
            cleaned = _remove_additional_properties(tools)
            cleaned = _remove_strict_from_schema(cleaned)

            key_after = cache.get_cache_key(
                model="hosted_vllm/test",
                messages=[{"role": "user", "content": "test"}],
                tools=tools,
            )

            assert key_before == key_after, (
                f"Cache key changed after sequential removal calls: "
                f"{key_before!r} != {key_after!r}"
            )


# ---------------------------------------------------------------------------
# In-place helpers: direct coverage
# ---------------------------------------------------------------------------

class TestInPlaceHelpers:
    """Direct tests for the _in_place internal helpers.
    These are called only on already-copied data, but we verify they
    actually perform the mutations they promise."""

    def test_remove_strict_in_place_deletes_strict(self):
        schema = {"strict": True, "properties": {"a": {"strict": True}}}

        _remove_strict_from_schema_in_place(schema)

        assert "strict" not in schema
        assert "strict" not in schema["properties"]["a"]

    def test_remove_strict_in_place_on_list(self):
        schemas = [{"strict": True}, {"strict": True, "nested": {"strict": True}}]

        _remove_strict_from_schema_in_place(schemas)

        assert "strict" not in schemas[0]
        assert "strict" not in schemas[1]
        assert "strict" not in schemas[1]["nested"]

    def test_remove_additional_properties_in_place_deletes(self):
        schema = {
            "additionalProperties": False,
            "properties": {"a": {"additionalProperties": False}},
        }

        _remove_additional_properties_in_place(schema)

        assert "additionalProperties" not in schema
        assert "additionalProperties" not in schema["properties"]["a"]

    def test_remove_additional_properties_in_place_keeps_true(self):
        """Only additionalProperties: false is removed; true is kept."""
        schema = {"additionalProperties": True}

        _remove_additional_properties_in_place(schema)

        assert schema["additionalProperties"] is True
