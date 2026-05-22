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
async def test_embedding_cache_preserves_prompt_tokens_details():
    """Test that prompt_tokens_details (including image_count) survives a full cache hit."""
    llm_caching_handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )

    cached_result = [
        {
            "embedding": [-0.025, -0.019],
            "index": 0,
            "object": "embedding",
            "model": "amazon.titan-embed-image-v1",
            "prompt_tokens_details": {"image_count": 1},
        }
    ]

    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = AsyncMock()
    response, cache_hit = llm_caching_handler._process_async_embedding_cached_response(
        final_embedding_cached_response=None,
        cached_result=cached_result,
        kwargs={"model": "amazon.titan-embed-image-v1", "input": "base64imagedata"},
        logging_obj=mock_logging_obj,
        start_time=datetime.now(),
        model="amazon.titan-embed-image-v1",
    )

    assert cache_hit
    assert response.usage is not None
    assert response.usage.prompt_tokens_details is not None
    assert response.usage.prompt_tokens_details.image_count == 1


@pytest.mark.asyncio
async def test_embedding_cache_backward_compat_no_prompt_tokens_details():
    """Test that old cached items without prompt_tokens_details still work."""
    llm_caching_handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )

    # Old-format cached item — no prompt_tokens_details field
    cached_result = [
        {
            "embedding": [-0.025, -0.019],
            "index": 0,
            "object": "embedding",
            "model": "text-embedding-ada-002",
        }
    ]

    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = AsyncMock()
    response, cache_hit = llm_caching_handler._process_async_embedding_cached_response(
        final_embedding_cached_response=None,
        cached_result=cached_result,
        kwargs={"model": "text-embedding-ada-002", "input": "test"},
        logging_obj=mock_logging_obj,
        start_time=datetime.now(),
        model="text-embedding-ada-002",
    )

    assert cache_hit
    assert response.usage is not None
    assert response.usage.prompt_tokens_details is None


@pytest.mark.asyncio
async def test_embedding_cache_aggregates_multiple_image_counts():
    """Test that image_count is summed correctly across multiple cached items."""
    llm_caching_handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )

    cached_result = [
        {
            "embedding": [-0.025, -0.019],
            "index": 0,
            "object": "embedding",
            "model": "amazon.titan-embed-image-v1",
            "prompt_tokens_details": {"image_count": 1},
        },
        {
            "embedding": [0.031, 0.042],
            "index": 1,
            "object": "embedding",
            "model": "amazon.titan-embed-image-v1",
            "prompt_tokens_details": {"image_count": 1},
        },
    ]

    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = AsyncMock()
    response, cache_hit = llm_caching_handler._process_async_embedding_cached_response(
        final_embedding_cached_response=None,
        cached_result=cached_result,
        kwargs={
            "model": "amazon.titan-embed-image-v1",
            "input": ["img1", "img2"],
        },
        logging_obj=mock_logging_obj,
        start_time=datetime.now(),
        model="amazon.titan-embed-image-v1",
    )

    assert cache_hit
    assert response.usage.prompt_tokens_details is not None
    assert response.usage.prompt_tokens_details.image_count == 2


def test_combine_usage_merges_prompt_tokens_details():
    """Test that combine_usage merges prompt_tokens_details from both Usage objects."""
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    llm_caching_handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )

    usage1 = Usage(
        prompt_tokens=10,
        completion_tokens=0,
        total_tokens=10,
        prompt_tokens_details=PromptTokensDetailsWrapper(image_count=1),
    )
    usage2 = Usage(
        prompt_tokens=20,
        completion_tokens=0,
        total_tokens=20,
        prompt_tokens_details=PromptTokensDetailsWrapper(image_count=2),
    )

    combined = llm_caching_handler.combine_usage(usage1, usage2)

    assert combined.prompt_tokens == 30
    assert combined.total_tokens == 30
    assert combined.prompt_tokens_details is not None
    assert combined.prompt_tokens_details.image_count == 3


def test_combine_usage_handles_none_details():
    """Test that combine_usage works when one or both sides have null prompt_tokens_details."""
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    llm_caching_handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )

    # Both null
    usage_a = Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)
    usage_b = Usage(prompt_tokens=20, completion_tokens=0, total_tokens=20)
    combined = llm_caching_handler.combine_usage(usage_a, usage_b)
    assert combined.prompt_tokens_details is None

    # Only first has details
    usage_c = Usage(
        prompt_tokens=10,
        completion_tokens=0,
        total_tokens=10,
        prompt_tokens_details=PromptTokensDetailsWrapper(image_count=1),
    )
    combined = llm_caching_handler.combine_usage(usage_c, usage_b)
    assert combined.prompt_tokens_details is not None
    assert combined.prompt_tokens_details.image_count == 1

    # Only second has details
    combined = llm_caching_handler.combine_usage(usage_a, usage_c)
    assert combined.prompt_tokens_details is not None
    assert combined.prompt_tokens_details.image_count == 1
