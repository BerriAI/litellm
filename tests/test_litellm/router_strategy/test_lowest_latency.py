#### What this tests ####
#    Latency values recorded by lowest-latency routing must be JSON
#    serializable for non-chat responses too (embeddings/speech/image skip
#    the ModelResponse branch, so the raw timedelta used to leak into the
#    latency list and break the Redis cache sync). Issue #33169.

import json
import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler

DEPLOYMENT_ID = "9876"
KWARGS = {
    "litellm_params": {
        "metadata": {
            "model_group": "gemini-embedding-001",
            "deployment": "vertex_ai/gemini-embedding-001",
        },
        "model_info": {"id": DEPLOYMENT_ID},
    }
}


def _embedding_response():
    return litellm.EmbeddingResponse(
        model="gemini-embedding-001",
        data=[{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
        object="list",
        usage=litellm.Usage(prompt_tokens=5, completion_tokens=0, total_tokens=5),
    )


def _recorded_latencies(cache: DualCache):
    cached = cache.get_cache(key="gemini-embedding-001_map") or {}
    return cached.get(DEPLOYMENT_ID, {}).get("latency", [])


def test_sync_embedding_latency_is_json_serializable():
    """log_success_event with datetime start/end (as the proxy passes) must not
    record a raw timedelta for non-ModelResponse results."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)

    start_time = datetime(2026, 1, 1, 12, 0, 0)
    end_time = datetime(2026, 1, 1, 12, 0, 2)

    handler.log_success_event(
        response_obj=_embedding_response(),
        kwargs=KWARGS,
        start_time=start_time,
        end_time=end_time,
    )

    latencies = _recorded_latencies(cache)
    assert latencies, "expected a latency entry to be recorded"
    assert all(
        not isinstance(value, timedelta) for value in latencies
    ), f"raw timedelta leaked into latency list: {latencies}"
    assert latencies[-1] == pytest.approx(2.0)
    # the exact failure mode from production: redis cache sync json.dumps
    json.dumps({"latency": latencies})


@pytest.mark.asyncio
async def test_async_embedding_latency_is_json_serializable():
    """async_log_success_event is the path the proxy actually hits."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)

    start_time = datetime(2026, 1, 1, 12, 0, 0)
    end_time = datetime(2026, 1, 1, 12, 0, 3)

    await handler.async_log_success_event(
        response_obj=_embedding_response(),
        kwargs=KWARGS,
        start_time=start_time,
        end_time=end_time,
    )

    latencies = _recorded_latencies(cache)
    assert latencies, "expected a latency entry to be recorded"
    assert all(
        not isinstance(value, timedelta) for value in latencies
    ), f"raw timedelta leaked into latency list: {latencies}"
    assert latencies[-1] == pytest.approx(3.0)
    json.dumps({"latency": latencies})


def _chat_response(completion_tokens: int):
    return litellm.ModelResponse(
        model="gpt-4o-mini",
        choices=[
            litellm.Choices(
                finish_reason="stop",
                index=0,
                message=litellm.Message(content="hi", role="assistant"),
            )
        ],
        usage=litellm.Usage(
            prompt_tokens=10,
            completion_tokens=completion_tokens,
            total_tokens=10 + completion_tokens,
        ),
    )


@pytest.mark.asyncio
async def test_async_chat_latency_normalized_per_token():
    """Chat responses go through the per-token normalization branch — with the
    up-front timedelta conversion the stored value must be seconds/token."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)

    await handler.async_log_success_event(
        response_obj=_chat_response(completion_tokens=4),
        kwargs=KWARGS,
        start_time=datetime(2026, 1, 1, 12, 0, 0),
        end_time=datetime(2026, 1, 1, 12, 0, 2),
    )

    latencies = _recorded_latencies(cache)
    assert latencies and latencies[-1] == pytest.approx(0.5)  # 2s / 4 tokens
    json.dumps({"latency": latencies})


@pytest.mark.asyncio
async def test_async_chat_zero_completion_tokens_falls_back_to_seconds():
    """safe_divide_seconds returns None for zero tokens — the fallback branch
    must store plain float seconds, not a timedelta."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)

    await handler.async_log_success_event(
        response_obj=_chat_response(completion_tokens=0),
        kwargs=KWARGS,
        start_time=datetime(2026, 1, 1, 12, 0, 0),
        end_time=datetime(2026, 1, 1, 12, 0, 3),
    )

    latencies = _recorded_latencies(cache)
    assert latencies and latencies[-1] == pytest.approx(3.0)
    assert not isinstance(latencies[-1], timedelta)
    json.dumps({"latency": latencies})


def test_sync_chat_zero_completion_tokens_falls_back_to_seconds():
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)

    handler.log_success_event(
        response_obj=_chat_response(completion_tokens=0),
        kwargs=KWARGS,
        start_time=datetime(2026, 1, 1, 12, 0, 0),
        end_time=datetime(2026, 1, 1, 12, 0, 2),
    )

    latencies = _recorded_latencies(cache)
    assert latencies and latencies[-1] == pytest.approx(2.0)
    assert not isinstance(latencies[-1], timedelta)
    json.dumps({"latency": latencies})
