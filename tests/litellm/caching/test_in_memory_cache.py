import asyncio
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock

from litellm.caching.in_memory_cache import InMemoryCache


def test_in_memory_openai_obj_cache():
    from openai import OpenAI

    openai_obj = OpenAI(api_key="my-fake-key")

    in_memory_cache = InMemoryCache()

    in_memory_cache.set_cache(key="my-fake-key", value=openai_obj)

    cached_obj = in_memory_cache.get_cache(key="my-fake-key")

    assert cached_obj is not None

    assert cached_obj == openai_obj


def test_in_memory_cache_max_size_per_item():
    """
    Test that the cache will not store items larger than the max size per item
    """
    in_memory_cache = InMemoryCache(max_size_per_item=100)

    result = in_memory_cache.check_value_size("a" * 100000000)

    assert result is False
