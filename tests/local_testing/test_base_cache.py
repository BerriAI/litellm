import os
import sys
import time
import traceback
import uuid

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import hashlib
import random

import pytest
from litellm.caching.base_cache import BaseCache


class TestCache(BaseCache):
    async def async_set_cache_pipeline(self, cache_list, **kwargs):
        pass


def test_get_ttl_with_maximum_ttl():
    """
    Test that get_ttl respects the maximum_ttl configuration when set
    """
    cache = TestCache(max_allowed_ttl=100)
    # Test with TTL exceeding maximum
    assert cache.get_ttl(ttl=150) == 100, "TTL should be capped at maximum_ttl"
    # Test with TTL below maximum
    assert (
        cache.get_ttl(ttl=50) == 50
    ), "TTL should remain unchanged when below maximum_ttl"
    # Test with TTL equal to maximum
    assert cache.get_ttl(ttl=100) == 100, "TTL should equal maximum_ttl when matching"


def test_get_ttl_without_maximum_ttl():
    """
    Test that get_ttl allows any TTL value when maximum_ttl is not set
    """
    cache = TestCache()  # No maximum_ttl set
    high_ttl = 1000000
    assert (
        cache.get_ttl(ttl=high_ttl) == high_ttl
    ), "TTL should not be capped when maximum_ttl is None"


def test_get_ttl_invalid_input():
    """
    Test that get_ttl handles invalid TTL inputs gracefully
    """
    cache = TestCache(default_ttl=60)
    # Test with non-integer TTL
    assert (
        cache.get_ttl(ttl="invalid") == 60
    ), "Should return default_ttl for invalid TTL input"
    # Test with no TTL specified
    assert cache.get_ttl() == 60, "Should return default_ttl when no TTL specified"
