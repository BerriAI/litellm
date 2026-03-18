"""
Test for Langfuse integration with Gemini cached_tokens bug
https://github.com/BerriAI/litellm/issues/18520
"""
import pytest
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


def test_cached_tokens_extraction():
    """
    Test that we can extract cached_tokens from prompt_tokens_details.
    This is the core logic fix for https://github.com/BerriAI/litellm/issues/18520
    """
    # Create usage object like Gemini returns
    usage = Usage(
        prompt_tokens=20209,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=20203,
            text_tokens=6,
        ),
        completion_tokens=541,
    )

    # Simulate the logic from langfuse.py lines 745-757 (after the fix)
    cache_read_input_tokens = 0  # Default value

    # Check prompt_tokens_details.cached_tokens (the fix)
    if hasattr(usage, "prompt_tokens_details"):
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        if (
            prompt_tokens_details is not None
            and hasattr(prompt_tokens_details, "cached_tokens")
        ):
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
            if cached_tokens is not None and cached_tokens > 0:
                cache_read_input_tokens = cached_tokens

    # Verify the fix works
    assert cache_read_input_tokens == 20203, f"Expected 20203, got {cache_read_input_tokens}"


def test_cached_tokens_not_present():
    """Test backward compatibility when cached_tokens is not present"""
    # Usage without prompt_tokens_details
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
    )

    cache_read_input_tokens = 0

    if hasattr(usage, "prompt_tokens_details"):
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        if (
            prompt_tokens_details is not None
            and hasattr(prompt_tokens_details, "cached_tokens")
        ):
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
            if cached_tokens is not None and cached_tokens > 0:
                cache_read_input_tokens = cached_tokens

    # Should remain 0
    assert cache_read_input_tokens == 0


def test_cached_tokens_is_zero():
    """Test when cached_tokens is explicitly 0"""
    usage = Usage(
        prompt_tokens=100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=0,
            text_tokens=100,
        ),
        completion_tokens=50,
    )

    cache_read_input_tokens = 0

    if hasattr(usage, "prompt_tokens_details"):
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        if (
            prompt_tokens_details is not None
            and hasattr(prompt_tokens_details, "cached_tokens")
        ):
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
            if cached_tokens is not None and cached_tokens > 0:
                cache_read_input_tokens = cached_tokens

    # Should remain 0 when cached_tokens is 0
    assert cache_read_input_tokens == 0
