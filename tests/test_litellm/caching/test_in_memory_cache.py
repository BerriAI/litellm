import asyncio
import json
import os
import sys
import time
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

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


def test_in_memory_cache_ttl():
    """
    Check that
    - if ttl is not set, it will be set to default ttl
    - if object expires, the ttl is also removed
    """
    in_memory_cache = InMemoryCache()

    in_memory_cache.set_cache(key="my-fake-key", value="my-fake-value", ttl=10)
    initial_ttl_time = in_memory_cache.ttl_dict["my-fake-key"]
    assert initial_ttl_time is not None

    in_memory_cache.set_cache(key="my-fake-key", value="my-fake-value-2", ttl=10)
    new_ttl_time = in_memory_cache.ttl_dict["my-fake-key"]
    assert new_ttl_time == initial_ttl_time  # ttl should not be updated

    ## On object expiration, the ttl should be removed
    in_memory_cache.set_cache(key="new-fake-key", value="new-fake-value", ttl=1)
    new_ttl_time = in_memory_cache.ttl_dict["new-fake-key"]
    assert new_ttl_time is not None
    time.sleep(1)
    cached_obj = in_memory_cache.get_cache(key="new-fake-key")
    new_ttl_time = in_memory_cache.ttl_dict.get("new-fake-key")
    assert new_ttl_time is None


def test_in_memory_cache_ttl_allow_override():
    """
    Check that
    - if ttl is not set, it will be set to default ttl
    - if object expires, the ttl is also removed
    """
    in_memory_cache = InMemoryCache()
    ## On object expiration, but no get_cache, the override should be allowed
    in_memory_cache.set_cache(key="new-fake-key", value="new-fake-value", ttl=1)
    initial_ttl_time = in_memory_cache.ttl_dict["new-fake-key"]
    assert initial_ttl_time is not None
    time.sleep(1)

    in_memory_cache.set_cache(key="new-fake-key", value="new-fake-value-2", ttl=1)
    new_ttl_time = in_memory_cache.ttl_dict["new-fake-key"]
    assert new_ttl_time is not None
    assert new_ttl_time != initial_ttl_time


@pytest.mark.asyncio
async def test_cache_cleanup_async_http_handler_with_event_loop():
    """Test that AsyncHTTPHandler cleanup works when an event loop is running"""
    in_memory_cache = InMemoryCache()
    
    # Create a mock AsyncHTTPHandler-like object
    mock_http_handler = MagicMock()
    mock_http_handler.client = MagicMock()
    mock_http_handler.client.is_closed = False
    mock_http_handler.close = AsyncMock()
    
    # Set up the cache with the mock handler
    in_memory_cache.set_cache(key="http_handler", value=mock_http_handler)
    
    # Mock get_running_loop to return a loop that supports create_task
    mock_loop = MagicMock()
    mock_loop.create_task = MagicMock()
    
    with patch('asyncio.get_running_loop', return_value=mock_loop):
        # Trigger cleanup by removing the key
        in_memory_cache._remove_key("http_handler")
    
    # Verify that create_task was called with the close coroutine
    mock_loop.create_task.assert_called_once()


@pytest.mark.asyncio 
async def test_cache_cleanup_async_http_handler_no_event_loop():
    """Test that AsyncHTTPHandler cleanup warns when no event loop is available"""
    in_memory_cache = InMemoryCache()
    
    # Create a mock AsyncHTTPHandler-like object
    mock_http_handler = MagicMock()
    mock_http_handler.client = MagicMock()
    mock_http_handler.client.is_closed = False
    mock_http_handler.close = AsyncMock()
    
    # Set up the cache with the mock handler
    in_memory_cache.set_cache(key="http_handler", value=mock_http_handler)
    
    # Mock get_running_loop to raise RuntimeError (no event loop)
    with patch('asyncio.get_running_loop', side_effect=RuntimeError("No event loop")):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            # Trigger cleanup by removing the key
            in_memory_cache._remove_key("http_handler")
            
            # Verify ResourceWarning was issued
            assert len(caught_warnings) == 1
            assert issubclass(caught_warnings[0].category, ResourceWarning)
            assert "HTTP client was evicted from cache but couldn't be closed properly" in str(caught_warnings[0].message)


def test_cache_cleanup_sync_http_handler():
    """Test that sync HTTPHandler cleanup works properly"""
    in_memory_cache = InMemoryCache()
    
    # Create a mock HTTPHandler-like object without is_closed attribute
    mock_http_handler = MagicMock()
    mock_http_handler.client = MagicMock()
    mock_http_handler.client.close = MagicMock()
    # Make sure client doesn't have is_closed to trigger sync handler path
    del mock_http_handler.client.is_closed
    
    # Set up the cache with the mock handler
    in_memory_cache.set_cache(key="http_handler", value=mock_http_handler)
    
    # Trigger cleanup by removing the key
    in_memory_cache._remove_key("http_handler")
    
    # Verify that the sync close method was called
    mock_http_handler.client.close.assert_called_once()


def test_cache_cleanup_non_http_object():
    """Test that cleanup doesn't affect non-HTTP objects"""
    in_memory_cache = InMemoryCache()
    
    # Create a regular object without HTTP client attributes
    regular_object = {"key": "value"}
    
    # Set up the cache with the regular object
    in_memory_cache.set_cache(key="regular_object", value=regular_object)
    
    # Trigger cleanup by removing the key - should not raise any errors
    in_memory_cache._remove_key("regular_object")
    
    # Verify the object was removed from cache
    assert in_memory_cache.get_cache("regular_object") is None


def test_cache_cleanup_exception_handling():
    """Test that cleanup exceptions are caught and don't break cache operations"""
    in_memory_cache = InMemoryCache()
    
    # Create a mock HTTP handler that raises an exception during cleanup
    mock_http_handler = MagicMock()
    mock_http_handler.client = MagicMock()
    mock_http_handler.client.close = MagicMock(side_effect=Exception("Cleanup error"))
    
    # Set up the cache with the mock handler
    in_memory_cache.set_cache(key="failing_handler", value=mock_http_handler)
    
    # Trigger cleanup by removing the key - should not raise any errors
    in_memory_cache._remove_key("failing_handler")
    
    # Verify the object was still removed from cache despite the exception
    assert in_memory_cache.get_cache("failing_handler") is None


def test_cache_eviction_triggers_cleanup():
    """Test that cache eviction properly triggers cleanup for HTTP clients"""
    in_memory_cache = InMemoryCache(max_size_in_memory=1)
    
    # Create mock HTTP handlers without is_closed attribute to trigger sync cleanup
    mock_handler1 = MagicMock()
    mock_handler1.client = MagicMock()
    mock_handler1.client.close = MagicMock()
    del mock_handler1.client.is_closed  # Remove is_closed to trigger sync cleanup path
    
    mock_handler2 = MagicMock()
    mock_handler2.client = MagicMock()
    mock_handler2.client.close = MagicMock()
    del mock_handler2.client.is_closed
    
    # Add first handler with short TTL
    in_memory_cache.set_cache(key="handler1", value=mock_handler1, ttl=1)
    
    # Wait for expiration
    time.sleep(1.1)
    
    # Add second handler, which should trigger eviction of expired handler1
    in_memory_cache.set_cache(key="handler2", value=mock_handler2)
    
    # Verify that handler1 was cleaned up during eviction
    mock_handler1.client.close.assert_called_once()
    
    # Verify handler2 is still in cache
    assert in_memory_cache.get_cache("handler2") is mock_handler2
