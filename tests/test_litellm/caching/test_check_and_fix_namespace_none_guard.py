"""
Test that check_and_fix_namespace handles None key gracefully.

Regression test for https://github.com/BerriAI/litellm/issues/30424
"""
from unittest.mock import MagicMock

from litellm.caching.redis_cache import RedisCache


def test_check_and_fix_namespace_with_none_key():
    """When key is None, check_and_fix_namespace should return None without raising."""
    cache = MagicMock(spec=RedisCache)
    cache.namespace = "litellm"
    # Call the real method
    result = RedisCache.check_and_fix_namespace(cache, key=None)
    assert result is None


def test_check_and_fix_namespace_with_none_key_no_namespace():
    """When key is None and namespace is None, should return None without raising."""
    cache = MagicMock(spec=RedisCache)
    cache.namespace = None
    result = RedisCache.check_and_fix_namespace(cache, key=None)
    assert result is None


def test_check_and_fix_namespace_with_valid_key():
    """Normal behavior: prefix key with namespace if not already prefixed."""
    cache = MagicMock(spec=RedisCache)
    cache.namespace = "litellm"
    result = RedisCache.check_and_fix_namespace(cache, key="my_key")
    assert result == "litellm:my_key"


def test_check_and_fix_namespace_with_already_prefixed_key():
    """If key already starts with namespace, don't double-prefix."""
    cache = MagicMock(spec=RedisCache)
    cache.namespace = "litellm"
    result = RedisCache.check_and_fix_namespace(cache, key="litellm:my_key")
    assert result == "litellm:my_key"


def test_check_and_fix_namespace_no_namespace():
    """When namespace is None, return key as-is."""
    cache = MagicMock(spec=RedisCache)
    cache.namespace = None
    result = RedisCache.check_and_fix_namespace(cache, key="my_key")
    assert result == "my_key"
