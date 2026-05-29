"""
Regression tests for the partial-cache-hit embedding index merge bug.

Exercises LLMCachingHandler._combine_cached_embedding_response_with_api_result
directly — no provider, no Redis, no event loop needed.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.caching.caching_handler import (
    CachingHandlerResponse,
    LLMCachingHandler,
)
from litellm.types.utils import Embedding, EmbeddingResponse


def _emb(index: int, marker: int) -> Embedding:
    """Build an Embedding with a recognisable marker so we can tell items apart."""
    return Embedding(embedding=[float(marker)], index=index, object="embedding")


def _emb_dict(index: int, marker: int) -> dict:
    """Dict variant — many providers (e.g. hosted_vllm) return dicts, not Embedding objects."""
    return {"embedding": [float(marker)], "index": index, "object": "embedding"}


def _run_merge(
    batch_size: int, cache_positions: set[int], provider_as_dict: bool = False
) -> list[int]:
    """
    Simulate what the handler builds at the cache-lookup step, then call the
    merge function with a synthetic provider response. Returns the resulting
    indices.
    """
    cached_data = []
    for pos in range(batch_size):
        if pos in cache_positions:
            cached_data.append(
                _emb(index=pos, marker=100 + pos)
            )  # cache hits use position-based index
        else:
            cached_data.append(None)
    cached_response = EmbeddingResponse(model="x", data=cached_data, object="list")

    num_misses = batch_size - len(cache_positions)
    make = _emb_dict if provider_as_dict else _emb
    provider_data = [make(index=i, marker=200 + i) for i in range(num_misses)]
    provider_response = EmbeddingResponse(model="x", data=provider_data, object="list")

    handler = LLMCachingHandler.__new__(LLMCachingHandler)
    chr_ = CachingHandlerResponse(final_embedding_cached_response=cached_response)

    merged = handler._combine_cached_embedding_response_with_api_result(
        _caching_handler_response=chr_,
        embedding_response=provider_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert merged is not None
    return [d["index"] if isinstance(d, dict) else d.index for d in merged.data]


def test_partial_cache_hit_two_in_the_middle():
    """File-31-style: 2 cache hits in the middle of a small batch."""
    actual = _run_merge(batch_size=8, cache_positions={1, 4})
    assert actual == [0, 1, 2, 3, 4, 5, 6, 7]


def test_partial_cache_hit_one_at_start():
    """One cache hit at the start shifts every following item by 1."""
    actual = _run_merge(batch_size=5, cache_positions={0})
    assert actual == [0, 1, 2, 3, 4]


def test_partial_cache_hit_no_duplicates():
    """All indices in the merged response must be unique."""
    for hits in ({1}, {1, 4}, {0, 2, 5}, set(range(15))):
        actual = _run_merge(batch_size=16, cache_positions=hits)
        assert len(set(actual)) == len(
            actual
        ), f"duplicate indices with hits={hits}: {actual}"


def test_all_cache_hits_no_provider_call_needed():
    """Sanity: every position is a cache hit, indices are correct."""
    actual = _run_merge(batch_size=4, cache_positions={0, 1, 2, 3})
    assert actual == [0, 1, 2, 3]


def test_no_cache_hits_provider_only():
    """Sanity: nothing cached, indices are 0..N-1 from the provider directly."""
    actual = _run_merge(batch_size=4, cache_positions=set())
    assert actual == [0, 1, 2, 3]


def test_real_world_file_31_pattern():
    """
    Exactly mirrors the observed production failure:
    128-element batch, cache hits at positions 16 and 17 produce drift +2 from pos 18.
    """
    actual = _run_merge(batch_size=128, cache_positions={16, 17})
    assert actual == list(range(128))


def test_provider_returns_dicts_not_embedding_objects():
    """
    Some providers (e.g. hosted_vllm) return raw dicts in EmbeddingResponse.data
    instead of Embedding pydantic instances. The merge must handle both.
    """
    actual = _run_merge(batch_size=8, cache_positions={1, 4}, provider_as_dict=True)
    assert actual == [0, 1, 2, 3, 4, 5, 6, 7]
