import pytest
from litellm.caching.caching import DualCache

def test_dual_cache_in_memory_set_get():
    cache = DualCache(in_memory_cache=True)
    cache.set_cache(key="user_spend_123", value=45.50, ttl=300)
    cached_val = cache.get_cache(key="user_spend_123")
    assert cached_val == 45.50

def test_dual_cache_miss():
    cache = DualCache(in_memory_cache=True)
    val = cache.get_cache(key="non_existent_key_xyz")
    assert val is None