"""
Test: OpenAI-style prompt_tokens_details.cached_tokens propagation to _cache_read_input_tokens.

Providers like Moonshot/Kimi and OpenAI (with prompt caching) return cached tokens in
prompt_tokens_details.cached_tokens. This should be propagated to _cache_read_input_tokens
for spend logging, streaming, and callback consistency.
"""
import pytest
from litellm.types.utils import Usage


def test_openai_style_cached_tokens_propagation():
    """cached_tokens in prompt_tokens_details should set _cache_read_input_tokens."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details={"cached_tokens": 800},
    )
    assert usage.prompt_tokens_details.cached_tokens == 800
    assert usage._cache_read_input_tokens == 800


def test_anthropic_style_still_works():
    """Anthropic's top-level cache_read_input_tokens should still work."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_input_tokens=800,
    )
    assert usage._cache_read_input_tokens == 800


def test_deepseek_style_still_works():
    """DeepSeek's prompt_cache_hit_tokens should still work."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_cache_hit_tokens=800,
    )
    assert usage._cache_read_input_tokens == 800


def test_no_double_counting():
    """When both Anthropic field and ptd.cached_tokens exist, no double-counting."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_input_tokens=800,
        prompt_tokens_details={"cached_tokens": 800},
    )
    assert usage._cache_read_input_tokens == 800  # Not 1600


def test_anthropic_takes_precedence():
    """When Anthropic field is set, ptd.cached_tokens doesn't override."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_input_tokens=600,
        prompt_tokens_details={"cached_tokens": 800},
    )
    assert usage._cache_read_input_tokens == 600  # Anthropic value wins


def test_no_cached_tokens():
    """When no cached tokens at all, _cache_read_input_tokens stays 0."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
    )
    assert usage._cache_read_input_tokens == 0


def test_cached_tokens_none():
    """When cached_tokens is None, don't propagate."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details={"cached_tokens": None},
    )
    assert usage._cache_read_input_tokens == 0


def test_moonshot_real_response():
    """Simulate actual Moonshot API response format."""
    usage = Usage(
        prompt_tokens=48502,
        completion_tokens=181,
        total_tokens=48683,
        prompt_tokens_details={"cached_tokens": 47872},
        completion_tokens_details={
            "reasoning_tokens": 100,
            "text_tokens": None,
            "audio_tokens": None,
        },
    )
    assert usage._cache_read_input_tokens == 47872
    assert usage.prompt_tokens_details.cached_tokens == 47872
