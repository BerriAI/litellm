"""
Tests that Anthropic Messages API-specific params (system, stop_sequences, top_k)
are included in cache key generation, preventing false cache hits.
"""

import pytest

from litellm.caching.caching import Cache
from litellm.litellm_core_utils.model_param_helper import ModelParamHelper


class TestAnthropicMessagesCacheKey:
    """Tests for cache key correctness with Anthropic-specific params."""

    def setup_method(self):
        self.cache = Cache()

    def test_anthropic_params_in_all_llm_api_params(self):
        """Verify system, stop_sequences, and top_k are in the combined param set."""
        all_params = ModelParamHelper._get_all_llm_api_params()
        assert "system" in all_params
        assert "stop_sequences" in all_params
        assert "top_k" in all_params

    def test_different_system_prompts_produce_different_cache_keys(self):
        """Two requests with different system prompts must NOT share a cache key."""
        key1 = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
        )
        key2 = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a pirate.",
        )
        assert key1 != key2

    def test_different_stop_sequences_produce_different_cache_keys(self):
        """Two requests with different stop_sequences must NOT share a cache key."""
        key1 = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["END"],
        )
        key2 = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["STOP", "DONE"],
        )
        assert key1 != key2

    def test_different_top_k_values_produce_different_cache_keys(self):
        """Two requests with different top_k values must NOT share a cache key."""
        key1 = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            top_k=10,
        )
        key2 = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            top_k=50,
        )
        assert key1 != key2

    def test_same_params_produce_same_cache_key(self):
        """Identical requests must produce the same cache key."""
        kwargs = dict(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
            stop_sequences=["END"],
            top_k=10,
        )
        key1 = self.cache.get_cache_key(**kwargs)
        key2 = self.cache.get_cache_key(**kwargs)
        assert key1 == key2

    def test_system_absent_vs_present_produces_different_cache_keys(self):
        """A request with system param vs without must produce different keys."""
        key_with_system = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
        )
        key_without_system = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert key_with_system != key_without_system

    def test_top_k_absent_vs_present_produces_different_cache_keys(self):
        """A request with top_k vs without must produce different keys."""
        key_with_top_k = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            top_k=40,
        )
        key_without_top_k = self.cache.get_cache_key(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert key_with_top_k != key_without_top_k
