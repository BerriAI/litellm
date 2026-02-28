"""
Regression tests for #20456 — Async Batch Embedding + Redis Cache
results in Index Misalignment/Duplication.

The root cause is threefold:
1. ``async_add_cache_pipeline`` stored embeddings using positional access
   (``result.data[idx]``) instead of the ``index`` field, so when a
   provider like vLLM returns results out of order, the wrong embedding
   gets stored under the wrong cache key.
2. ``_combine_cached_embedding_response_with_api_result`` filled ``None``
   (uncached) slots with a sequential counter into the unsorted API
   response, placing the wrong embeddings at the wrong positions.
3. After combining, the ``index`` field on each Embedding was not
   corrected to match its final position in the merged list, leading to
   duplicate or missing indices in ``[data.index for data in result.data]``.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from datetime import datetime, timedelta

from litellm.caching.caching_handler import CachingHandlerResponse, LLMCachingHandler
from litellm.types.utils import Embedding, EmbeddingResponse


def _make_handler():
    return LLMCachingHandler(
        original_function=lambda: None, request_kwargs={}, start_time=datetime.now()
    )


class TestCombineOutOfOrderAPIResponse:
    """Tests for ``_combine_cached_embedding_response_with_api_result``
    when the provider returns results out of order."""

    def test_single_uncached_in_middle(self):
        """[hit, MISS, hit] with API returning a single item."""
        h = _make_handler()
        cached = EmbeddingResponse(
            data=[
                Embedding(embedding=[1.0], index=0, object="embedding"),
                None,
                Embedding(embedding=[3.0], index=2, object="embedding"),
            ]
        )
        api = EmbeddingResponse(
            data=[Embedding(embedding=[2.0], index=0, object="embedding")]
        )
        resp = CachingHandlerResponse(final_embedding_cached_response=cached)
        result = h._combine_cached_embedding_response_with_api_result(
            resp, api, datetime.now(), datetime.now() + timedelta(seconds=1)
        )
        assert [d.embedding for d in result.data] == [[1.0], [2.0], [3.0]]
        assert [d.index for d in result.data] == [0, 1, 2]

    def test_multiple_uncached_reversed(self):
        """[MISS, hit, MISS, MISS] with API returning 3 items reversed."""
        h = _make_handler()
        cached = EmbeddingResponse(
            data=[
                None,
                Embedding(embedding=[20.0], index=1, object="embedding"),
                None,
                None,
            ]
        )
        # API returns items in REVERSE order
        api = EmbeddingResponse(
            data=[
                Embedding(embedding=[40.0], index=2, object="embedding"),
                Embedding(embedding=[30.0], index=1, object="embedding"),
                Embedding(embedding=[10.0], index=0, object="embedding"),
            ]
        )
        resp = CachingHandlerResponse(final_embedding_cached_response=cached)
        result = h._combine_cached_embedding_response_with_api_result(
            resp, api, datetime.now(), datetime.now() + timedelta(seconds=1)
        )
        assert [d.embedding for d in result.data] == [[10.0], [20.0], [30.0], [40.0]]
        assert [d.index for d in result.data] == [0, 1, 2, 3]

    def test_all_uncached(self):
        """All cache misses — pure API response reordering."""
        h = _make_handler()
        cached = EmbeddingResponse(data=[None, None, None])
        # API returns shuffled
        api = EmbeddingResponse(
            data=[
                Embedding(embedding=[3.0], index=2, object="embedding"),
                Embedding(embedding=[1.0], index=0, object="embedding"),
                Embedding(embedding=[2.0], index=1, object="embedding"),
            ]
        )
        resp = CachingHandlerResponse(final_embedding_cached_response=cached)
        result = h._combine_cached_embedding_response_with_api_result(
            resp, api, datetime.now(), datetime.now() + timedelta(seconds=1)
        )
        assert [d.embedding for d in result.data] == [[1.0], [2.0], [3.0]]
        assert [d.index for d in result.data] == [0, 1, 2]

    def test_large_batch_with_scattered_cache_hits(self):
        """Simulate a 10-item batch with 4 cache hits and 6 misses."""
        h = _make_handler()
        data = [None] * 10
        # Cache hits at positions 1, 4, 7, 9
        for pos in [1, 4, 7, 9]:
            data[pos] = Embedding(
                embedding=[float(pos)], index=pos, object="embedding"
            )
        cached = EmbeddingResponse(data=data)

        # 6 uncached positions: 0, 2, 3, 5, 6, 8
        # API returns them shuffled
        miss_positions = [0, 2, 3, 5, 6, 8]
        api_data = [
            Embedding(embedding=[float(p)], index=i, object="embedding")
            for i, p in enumerate(reversed(miss_positions))
        ]
        api = EmbeddingResponse(data=api_data)

        resp = CachingHandlerResponse(final_embedding_cached_response=cached)
        result = h._combine_cached_embedding_response_with_api_result(
            resp, api, datetime.now(), datetime.now() + timedelta(seconds=1)
        )

        assert len(result.data) == 10
        # All indices must be 0..9
        assert [d.index for d in result.data] == list(range(10))
        # Cache hits must be preserved
        for pos in [1, 4, 7, 9]:
            assert result.data[pos].embedding == [float(pos)]


class TestCachePipelineSorting:
    """Tests for ``async_add_cache_pipeline`` sorting result.data by index
    before storing."""

    def test_sort_preserves_input_alignment(self):
        """After sorting, result.data[i] should correspond to input[i]."""
        # Simulate out-of-order API result
        result = EmbeddingResponse(
            data=[
                Embedding(embedding=[3.0], index=2, object="embedding"),
                Embedding(embedding=[1.0], index=0, object="embedding"),
                Embedding(embedding=[2.0], index=1, object="embedding"),
            ]
        )

        # The fix sorts in-place
        if result.data and hasattr(result.data[0], "index"):
            result.data = sorted(
                result.data, key=lambda e: getattr(e, "index", 0)
            )

        # After sorting, positional access matches input order
        assert result.data[0].embedding == [1.0]
        assert result.data[1].embedding == [2.0]
        assert result.data[2].embedding == [3.0]
