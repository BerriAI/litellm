"""
Tests for _transform_response_api_usage_to_chat_usage in litellm/responses/utils.py.

Covers the cache_write_tokens -> cache_creation_tokens mapping in both the dict
path and the object (getattr) path, plus the no-cache-write and None cases.
"""

import pytest
from unittest.mock import MagicMock

from litellm.types.llms.openai import ResponseAPIUsage
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


# ---------------------------------------------------------------------------
# Helper: import the function under test
# ---------------------------------------------------------------------------

def _get_transform_fn():
    from litellm.responses.utils import ResponsesAPIRequestUtils
    return ResponsesAPIRequestUtils._transform_response_api_usage_to_chat_usage


# ===================================================================
# 1. Dict path -- cache_write_tokens present
#    Expected: cache_write_tokens is mapped to cache_creation_tokens
# ===================================================================

class TestDictPathWithCacheWriteTokens:
    def test_cache_write_tokens_mapped_to_cache_creation_tokens(self):
        """When input_tokens_details is a dict containing cache_write_tokens,
        it should be remapped to cache_creation_tokens on the resulting
        PromptTokensDetailsWrapper."""
        transform = _get_transform_fn()

        usage_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {
                "cached_tokens": 30,
                "text_tokens": 70,
                "cache_write_tokens": 25,
            },
        }

        result: Usage = transform(usage_dict)

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.prompt_tokens_details is not None
        # cache_write_tokens should have been remapped to cache_creation_tokens
        assert result.prompt_tokens_details.cache_creation_tokens == 25
        # cached_tokens should be preserved
        assert result.prompt_tokens_details.cached_tokens == 30
        # text_tokens should be preserved
        assert result.prompt_tokens_details.text_tokens == 70

    def test_cache_write_tokens_not_overwriting_existing_cache_creation(self):
        """When both cache_write_tokens and cache_creation_tokens are present
        in the dict, cache_creation_tokens should NOT be overwritten."""
        transform = _get_transform_fn()

        usage_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {
                "cached_tokens": 30,
                "cache_write_tokens": 25,
                "cache_creation_tokens": 40,  # already present
            },
        }

        result: Usage = transform(usage_dict)

        assert result.prompt_tokens_details is not None
        # cache_creation_tokens should keep its original value (40), not be overwritten by 25
        assert result.prompt_tokens_details.cache_creation_tokens == 40

    def test_original_dict_not_mutated(self):
        """The caller's dict must not be mutated by the transform."""
        transform = _get_transform_fn()

        input_details = {
            "cached_tokens": 30,
            "cache_write_tokens": 25,
        }
        usage_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": input_details,
        }

        transform(usage_dict)

        # cache_write_tokens should still be in the original dict
        assert "cache_write_tokens" in input_details
        assert "cache_creation_tokens" not in input_details


# ===================================================================
# 2. Object path -- cache_write_tokens present
#    Expected: cache_write_tokens is mapped to cache_creation_tokens
# ===================================================================

class TestObjectPathWithCacheWriteTokens:
    def test_cache_write_tokens_mapped_to_cache_creation_tokens(self):
        """When input_tokens_details is an object with cache_write_tokens,
        it should be mapped to cache_creation_tokens on the resulting
        PromptTokensDetailsWrapper."""
        transform = _get_transform_fn()

        # Build a ResponseAPIUsage with an InputTokensDetails object that has
        # cache_write_tokens as an extra field (model_config = {"extra": "allow"})
        usage_input = ResponseAPIUsage(
            input_tokens=200,
            output_tokens=80,
            total_tokens=280,
            input_tokens_details={
                "cached_tokens": 50,
                "text_tokens": 150,
                "cache_write_tokens": 60,
            },
        )

        result: Usage = transform(usage_input)

        assert result.prompt_tokens == 200
        assert result.completion_tokens == 80
        assert result.total_tokens == 280
        assert result.prompt_tokens_details is not None
        # cache_write_tokens should be mapped to cache_creation_tokens
        assert result.prompt_tokens_details.cache_creation_tokens == 60
        # cached_tokens should be preserved
        assert result.prompt_tokens_details.cached_tokens == 50
        # text_tokens should be preserved
        assert result.prompt_tokens_details.text_tokens == 150

    def test_object_path_with_mock_input_tokens_details(self):
        """Use a mock object for input_tokens_details to directly control
        getattr behaviour, verifying the object path independently of
        Pydantic's extra-field handling."""
        transform = _get_transform_fn()

        mock_details = MagicMock()
        mock_details.cache_write_tokens = 42
        mock_details.cached_tokens = 10
        mock_details.audio_tokens = None
        mock_details.text_tokens = 100
        mock_details.image_tokens = None

        usage_input = ResponseAPIUsage(
            input_tokens=150,
            output_tokens=60,
            total_tokens=210,
        )
        # Override input_tokens_details with our mock (bypass Pydantic validation)
        usage_input.input_tokens_details = mock_details

        result: Usage = transform(usage_input)

        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.cache_creation_tokens == 42
        assert result.prompt_tokens_details.cached_tokens == 10
        assert result.prompt_tokens_details.text_tokens == 100

    def test_object_path_no_cache_write_tokens(self):
        """When the object has no cache_write_tokens attribute (returns None),
        cache_creation_tokens should not be set."""
        transform = _get_transform_fn()

        mock_details = MagicMock()
        mock_details.cache_write_tokens = None
        mock_details.cached_tokens = 20
        mock_details.audio_tokens = None
        mock_details.text_tokens = 80
        mock_details.image_tokens = None

        usage_input = ResponseAPIUsage(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
        )
        usage_input.input_tokens_details = mock_details

        result: Usage = transform(usage_input)

        assert result.prompt_tokens_details is not None
        # cache_creation_tokens should not be set (None -> deleted in __init__)
        assert not hasattr(result.prompt_tokens_details, "cache_creation_tokens")
        assert result.prompt_tokens_details.cached_tokens == 20


# ===================================================================
# 3. Dict path -- no cache_write_tokens
#    Expected: no cache_creation_tokens mapping occurs
# ===================================================================

class TestDictPathWithoutCacheWriteTokens:
    def test_no_cache_write_tokens_no_mapping(self):
        """When input_tokens_details dict has no cache_write_tokens,
        cache_creation_tokens should not appear on the result."""
        transform = _get_transform_fn()

        usage_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {
                "cached_tokens": 30,
                "text_tokens": 70,
            },
        }

        result: Usage = transform(usage_dict)

        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.cached_tokens == 30
        assert result.prompt_tokens_details.text_tokens == 70
        # cache_creation_tokens should not be set
        assert not hasattr(result.prompt_tokens_details, "cache_creation_tokens")

    def test_existing_cache_creation_tokens_preserved(self):
        """When cache_creation_tokens is already in the dict (no cache_write_tokens),
        it should be preserved as-is."""
        transform = _get_transform_fn()

        usage_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {
                "cached_tokens": 30,
                "cache_creation_tokens": 15,
            },
        }

        result: Usage = transform(usage_dict)

        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.cache_creation_tokens == 15


# ===================================================================
# 4. None case -- usage_input is None
#    Expected: returns Usage with all zeros
# ===================================================================

class TestNoneUsageInput:
    def test_none_returns_zero_usage(self):
        """When usage_input is None, should return Usage with all zero fields."""
        transform = _get_transform_fn()

        result: Usage = transform(None)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        assert result.prompt_tokens_details is None
        assert result.completion_tokens_details is None


# ===================================================================
# 5. Additional edge cases
# ===================================================================

class TestEdgeCases:
    def test_total_tokens_auto_calculated_when_missing(self):
        """When total_tokens is not provided, it should be computed as
        input_tokens + output_tokens."""
        transform = _get_transform_fn()

        usage_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_tokens_details": {
                "cached_tokens": 30,
                "cache_write_tokens": 20,
            },
        }

        result: Usage = transform(usage_dict)

        assert result.total_tokens == 150

    def test_output_tokens_details_mapped(self):
        """Verify output_tokens_details are correctly mapped to
        completion_tokens_details."""
        transform = _get_transform_fn()

        usage_input = ResponseAPIUsage(
            input_tokens=100,
            output_tokens=80,
            total_tokens=180,
            output_tokens_details={
                "reasoning_tokens": 30,
                "text_tokens": 50,
            },
        )

        result: Usage = transform(usage_input)

        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.reasoning_tokens == 30
        assert result.completion_tokens_details.text_tokens == 50

    def test_cost_preserved(self):
        """When the ResponseAPIUsage has a cost, it should be preserved on the result."""
        transform = _get_transform_fn()

        usage_input = ResponseAPIUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.0042,
        )

        result: Usage = transform(usage_input)

        assert result.cost == 0.0042
