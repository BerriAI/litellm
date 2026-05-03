"""
Tests that LowestLatencyLoggingHandler stores latency values as floats
(not raw datetime.timedelta) in the cache, across all success-callback branches.

Regression test for: timedelta values in the {model_group}_map cache breaking
RedisCache.async_set_cache JSON serialization. Earlier partial fix in PR #14040
covered only the sync ModelResponse-with-usage branch.
"""

import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.types.utils import EmbeddingResponse


def _make_kwargs(deployment_id: str = "1234", model_group: str = "embed-group") -> dict:
    return {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": "openai/text-embedding-3-small",
            },
            "model_info": {"id": deployment_id},
        }
    }


def _latency_list(cache: DualCache, model_group: str, deployment_id: str) -> list:
    cached = cache.get_cache(key=f"{model_group}_map")
    assert cached is not None
    assert deployment_id in cached
    return cached[deployment_id].get("latency", [])


def test_log_success_event_embedding_response_stores_float():
    """EmbeddingResponse hits the non-ModelResponse branch — final_value must be float."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)
    kwargs = _make_kwargs()
    response_obj = EmbeddingResponse(
        model="text-embedding-3-small",
        usage=litellm.Usage(prompt_tokens=4, completion_tokens=0, total_tokens=4),
    )
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1.5)

    handler.log_success_event(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=start_time,
        end_time=end_time,
    )

    latencies = _latency_list(cache, "embed-group", "1234")
    assert latencies, "expected one latency entry"
    assert all(isinstance(v, float) for v in latencies), f"non-float in {latencies!r}"


def test_log_success_event_model_response_no_usage_stores_float():
    """ModelResponse with usage=None hits a sub-branch that skips conversion — must still produce float."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)
    kwargs = _make_kwargs(model_group="chat-group")
    response_obj = litellm.ModelResponse()
    response_obj.usage = None  # type: ignore[attr-defined]
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=0.75)

    handler.log_success_event(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=start_time,
        end_time=end_time,
    )

    latencies = _latency_list(cache, "chat-group", "1234")
    assert latencies and all(isinstance(v, float) for v in latencies)


@pytest.mark.asyncio
async def test_async_log_success_event_embedding_response_stores_float():
    """Async mirror of the embedding test."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)
    kwargs = _make_kwargs()
    response_obj = EmbeddingResponse(
        model="text-embedding-3-small",
        usage=litellm.Usage(prompt_tokens=4, completion_tokens=0, total_tokens=4),
    )
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=2.0)

    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=start_time,
        end_time=end_time,
    )

    latencies = _latency_list(cache, "embed-group", "1234")
    assert latencies and all(isinstance(v, float) for v in latencies)


@pytest.mark.asyncio
async def test_async_log_success_event_zero_completion_tokens_stores_float():
    """Async ModelResponse with completion_tokens=0 takes the safe_divide_seconds → None
    fallback at lowest_latency.py:344, which PR #14040 only fixed in the sync mirror."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)
    kwargs = _make_kwargs(model_group="chat-group")
    response_obj = litellm.ModelResponse(
        usage=litellm.Usage(prompt_tokens=100, completion_tokens=0, total_tokens=100),
    )
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=0.5)

    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=start_time,
        end_time=end_time,
    )

    latencies = _latency_list(cache, "chat-group", "1234")
    assert latencies and all(isinstance(v, float) for v in latencies)
