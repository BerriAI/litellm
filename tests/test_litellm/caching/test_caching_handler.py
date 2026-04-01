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


# ---------------------------------------------------------------------------
# Tests for deferred model_dump_json() in async_set_cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_set_cache_defers_model_dump_json():
    """
    async_set_cache must NOT call model_dump_json() before create_task —
    the serialization must happen inside the task body so the large JSON
    string is not held in memory while the task waits in the queue.
    Regression test for Vertex image generation ~479MB peak memory spike.
    """
    import litellm
    from litellm.caching.caching import Cache
    from litellm.types.utils import ModelResponse, Usage

    # Build a minimal ModelResponse
    result = ModelResponse(
        id="test-id",
        model="gemini-2.0-flash",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    call_order = []

    original_model_dump_json = result.model_dump_json

    def tracking_model_dump_json():
        call_order.append("model_dump_json")
        return original_model_dump_json()

    result.model_dump_json = tracking_model_dump_json

    # Patch litellm.cache with a mock that captures the serialized value
    mock_cache = MagicMock(spec=Cache)
    captured = {}

    async def fake_async_add_cache(serialized, **kwargs):
        captured["value"] = serialized
        call_order.append("async_add_cache")

    mock_cache.async_add_cache = fake_async_add_cache
    mock_cache.cache = MagicMock()  # not S3Cache

    llm_caching_handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )
    llm_caching_handler.dual_cache = MagicMock()

    with patch("litellm.cache", mock_cache):
        with patch.object(llm_caching_handler, "_should_store_result_in_cache", return_value=True):
            await llm_caching_handler.async_set_cache(
                result=result,
                original_function=MagicMock(),
                kwargs={"model": "gemini-2.0-flash"},
                args=None,
            )
            # model_dump_json must NOT have been called yet (before task runs)
            assert "model_dump_json" not in call_order, (
                "model_dump_json() was called eagerly before the task ran — "
                "this causes the large JSON string to be held in memory during queue wait"
            )
            # Now run the event loop to execute the pending task
            await asyncio.sleep(0)

    # After task execution both calls should have happened
    assert "model_dump_json" in call_order
    assert "async_add_cache" in call_order
    # model_dump_json must be called BEFORE async_add_cache (inside the task)
    assert call_order.index("model_dump_json") < call_order.index("async_add_cache")


@pytest.mark.asyncio
async def test_async_set_cache_skips_oversized_responses():
    """
    async_set_cache must NOT call async_add_cache when model_dump_json()
    produces a string larger than MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB * 1024.

    The early return must happen before any `await`, ensuring the large string
    is freed immediately and does not accumulate across N concurrent tasks.
    Regression test for Vertex image generation ~803MB peak from 64 concurrent
    _cache_result tasks each holding a ~12.5 MB serialized string during I/O.
    """
    import litellm
    from litellm.caching.caching import Cache
    from litellm.constants import MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB
    from litellm.types.utils import ModelResponse, Usage

    result = ModelResponse(
        id="oversized-id",
        model="gemini-2.0-flash",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    # Return a string that exceeds the per-item limit
    oversized_json = "x" * (MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB * 1024 + 1)
    result.model_dump_json = lambda: oversized_json  # type: ignore[method-assign]

    cache_called = []

    async def fake_async_add_cache(serialized, **kwargs):
        cache_called.append(serialized)

    mock_cache = MagicMock(spec=Cache)
    mock_cache.async_add_cache = fake_async_add_cache
    mock_cache.cache = MagicMock()

    llm_caching_handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )
    llm_caching_handler.dual_cache = MagicMock()

    with patch("litellm.cache", mock_cache):
        with patch.object(
            llm_caching_handler, "_should_store_result_in_cache", return_value=True
        ):
            await llm_caching_handler.async_set_cache(
                result=result,
                original_function=MagicMock(),
                kwargs={"model": "gemini-2.0-flash"},
                args=None,
            )
            await asyncio.sleep(0)  # drain pending tasks

    assert not cache_called, (
        "async_add_cache was called for an oversized response — "
        "this would hold the large serialized string in memory during cache I/O"
    )
