import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.openai.common_utils import BaseOpenAILLM


def test_openai_client_reuse_sync():
    """
    Test that multiple synchronous completion calls reuse the same OpenAI client
    """
    litellm.set_verbose = True

    # Mock the OpenAI client creation to track how many times it's called
    with patch("litellm.llms.openai.openai.OpenAI") as mock_openai, patch.object(
        BaseOpenAILLM, "set_cached_openai_client"
    ) as mock_set_cache, patch.object(
        BaseOpenAILLM, "get_cached_openai_client"
    ) as mock_get_cache:

        # Setup the mock to return None first time (cache miss) then a client for subsequent calls
        mock_client = MagicMock()
        mock_get_cache.side_effect = [None] + [
            mock_client
        ] * 9  # First call returns None, rest return the mock client

        # Make 10 completion calls
        for _ in range(10):
            try:
                litellm.completion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                )
            except Exception:
                # We expect exceptions since we're mocking the client
                pass

        # Verify OpenAI client was created only once
        assert mock_openai.call_count == 1, "OpenAI client should be created only once"

        # Verify the client was cached
        assert mock_set_cache.call_count == 1, "Client should be cached once"

        # Verify we tried to get from cache 10 times (once per request)
        assert mock_get_cache.call_count == 10, "Should check cache for each request"


@pytest.mark.asyncio
async def test_openai_client_reuse_async():
    """
    Test that multiple asynchronous completion calls reuse the same OpenAI client
    """
    litellm.set_verbose = True

    # Mock the AsyncOpenAI client creation to track how many times it's called
    with patch(
        "litellm.llms.openai.openai.AsyncOpenAI"
    ) as mock_async_openai, patch.object(
        BaseOpenAILLM, "set_cached_openai_client"
    ) as mock_set_cache, patch.object(
        BaseOpenAILLM, "get_cached_openai_client"
    ) as mock_get_cache:

        # Setup the mock to return None first time (cache miss) then a client for subsequent calls
        mock_client = MagicMock()
        mock_get_cache.side_effect = [None] + [
            mock_client
        ] * 9  # First call returns None, rest return the mock client

        # Make 10 async completion calls
        for _ in range(10):
            try:
                await litellm.acompletion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                )
            except Exception:
                # We expect exceptions since we're mocking the client
                pass

        # Verify AsyncOpenAI client was created only once
        assert (
            mock_async_openai.call_count == 1
        ), "AsyncOpenAI client should be created only once"

        # Verify the client was cached
        assert mock_set_cache.call_count == 1, "Client should be cached once"

        # Verify we tried to get from cache 10 times (once per request)
        assert mock_get_cache.call_count == 10, "Should check cache for each request"


@pytest.mark.asyncio
async def test_openai_client_reuse_streaming():
    """
    Test that multiple streaming completion calls reuse the same OpenAI client
    """
    litellm.set_verbose = True

    # Mock the AsyncOpenAI client creation to track how many times it's called
    with patch(
        "litellm.llms.openai.openai.AsyncOpenAI"
    ) as mock_async_openai, patch.object(
        BaseOpenAILLM, "set_cached_openai_client"
    ) as mock_set_cache, patch.object(
        BaseOpenAILLM, "get_cached_openai_client"
    ) as mock_get_cache:

        # Setup the mock to return None first time (cache miss) then a client for subsequent calls
        mock_client = MagicMock()
        mock_get_cache.side_effect = [None] + [
            mock_client
        ] * 9  # First call returns None, rest return the mock client

        # Make 10 streaming completion calls
        for _ in range(10):
            try:
                await litellm.acompletion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                    stream=True,
                )
            except Exception:
                # We expect exceptions since we're mocking the client
                pass

        # Verify AsyncOpenAI client was created only once
        assert (
            mock_async_openai.call_count == 1
        ), "AsyncOpenAI client should be created only once"

        # Verify the client was cached
        assert mock_set_cache.call_count == 1, "Client should be cached once"

        # Verify we tried to get from cache 10 times (once per request)
        assert mock_get_cache.call_count == 10, "Should check cache for each request"


def test_openai_client_reuse_with_different_params():
    """
    Test that different client parameters create different cached clients
    """
    litellm.set_verbose = True

    # Mock the OpenAI client creation
    with patch("litellm.llms.openai.openai.OpenAI") as mock_openai, patch.object(
        BaseOpenAILLM, "set_cached_openai_client"
    ) as mock_set_cache, patch.object(
        BaseOpenAILLM, "get_cached_openai_client", return_value=None
    ) as mock_get_cache:

        # Make calls with different API keys
        try:
            litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                api_key="test_key_1",
            )
        except Exception:
            pass

        try:
            litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                api_key="test_key_2",
            )
        except Exception:
            pass

        # Verify OpenAI client was created twice (different API keys)
        assert (
            mock_openai.call_count == 2
        ), "Different API keys should create different clients"

        # Verify the clients were cached
        assert mock_set_cache.call_count == 2, "Both clients should be cached"

        # Verify we tried to get from cache twice
        assert mock_get_cache.call_count == 2, "Should check cache for each request"


def test_openai_client_reuse_with_custom_client():
    """
    Test that when a custom client is provided, it's used directly without caching
    """
    litellm.set_verbose = True

    # Create a mock custom client
    custom_client = MagicMock()

    # Mock the cache functions
    with patch.object(
        BaseOpenAILLM, "set_cached_openai_client"
    ) as mock_set_cache, patch.object(
        BaseOpenAILLM, "get_cached_openai_client"
    ) as mock_get_cache:

        # Make multiple calls with the custom client
        for _ in range(5):
            try:
                litellm.completion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    client=custom_client,
                )
            except Exception:
                pass

        # Verify we never tried to cache the client
        assert mock_set_cache.call_count == 0, "Custom client should not be cached"

        # Verify we never tried to get from cache
        assert (
            mock_get_cache.call_count == 0
        ), "Should not check cache when custom client is provided"
