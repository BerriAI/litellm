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
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from litellm.caching.caching_handler import LLMCachingHandler


@pytest.mark.asyncio
async def test_process_async_embedding_cached_response():
    llm_caching_handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )

    args = {
        "cached_result": [
            {
                "embedding": [-0.025122925639152527, -0.019487135112285614],
                "index": 0,
                "object": "embedding",
            }
        ]
    }

    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = AsyncMock()
    response, cache_hit = llm_caching_handler._process_async_embedding_cached_response(
        final_embedding_cached_response=None,
        cached_result=args["cached_result"],
        kwargs={"model": "text-embedding-ada-002", "input": "test"},
        logging_obj=mock_logging_obj,
        start_time=datetime.now(),
        model="text-embedding-ada-002",
    )

    assert cache_hit

    print(f"response: {response}")
    assert len(response.data) == 1


@pytest.mark.asyncio
async def test_async_generate_streaming_content_uses_asyncio_sleep():
    """should yield ModelResponseStream chunks using asyncio.sleep (non-blocking)."""
    from litellm.caching.caching import Cache
    from litellm.types.utils import ModelResponseStream

    cache = Cache()
    content = "Hello world!"
    chunks = []
    async for chunk in cache.async_generate_streaming_content(content):
        chunks.append(chunk)

    assert all(
        isinstance(c, ModelResponseStream) for c in chunks
    ), "All chunks must be ModelResponseStream objects, not plain dicts"
    full = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert full == content
    assert len(chunks) > 1, "should produce multiple chunks"


@pytest.mark.asyncio
async def test_convert_cached_stream_response_uses_async_generate_for_text_content():
    """should call async_generate_streaming_content when litellm.cache is set
    and the cached result contains simple text content (no tool calls)."""
    import litellm
    from litellm.caching.caching import Cache
    from litellm.caching.caching_handler import LLMCachingHandler

    handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )

    cached_result = {
        "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}]
    }

    mock_cache = MagicMock(spec=Cache)
    mock_cache.async_generate_streaming_content = MagicMock(return_value=aiter_mock([]))

    logging_obj = MagicMock()

    with patch.object(litellm, "cache", mock_cache):
        result = handler._convert_cached_stream_response(
            cached_result=cached_result,
            call_type="acompletion",
            logging_obj=logging_obj,
            model="gpt-4o",
        )

    mock_cache.async_generate_streaming_content.assert_called_once_with("Hi there!")


@pytest.mark.asyncio
async def test_convert_cached_stream_response_falls_back_for_tool_calls():
    """should fall back to convert_to_streaming_response_async when cached result
    has tool calls (not simple text content)."""
    import litellm
    from litellm.caching.caching import Cache
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_streaming_response_async,
    )

    handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )

    cached_result = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "get_weather"}}
                    ],
                }
            }
        ]
    }

    mock_cache = MagicMock(spec=Cache)
    logging_obj = MagicMock()

    with (
        patch.object(litellm, "cache", mock_cache),
        patch("litellm.utils.convert_to_streaming_response_async") as mock_fallback,
    ):
        mock_fallback.return_value = aiter_mock([])
        handler._convert_cached_stream_response(
            cached_result=cached_result,
            call_type="acompletion",
            logging_obj=logging_obj,
            model="gpt-4o",
        )

    mock_cache.async_generate_streaming_content.assert_not_called()
    mock_fallback.assert_called_once_with(response_object=cached_result)


async def aiter_mock(items):
    """Helper: async generator that yields items from a list."""
    for item in items:
        yield item
