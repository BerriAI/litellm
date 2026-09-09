#### What this tests ####
#    Latency values recorded by lowest-latency routing must be JSON
#    serializable for non-chat responses too (embeddings/speech/image skip
#    the ModelResponse branch, so the raw timedelta used to leak into the
#    latency list and break the Redis cache sync). Issue #33169.

import json
from datetime import datetime, timedelta

import pytest


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


MODEL_GROUP = "gpt-4o-mini"
FAST_TTFT_ID = "fast-ttft-short-output"
SLOW_TTFT_ID = "slow-ttft-long-output"
STREAMING_DEPLOYMENTS = [
    {"model_info": {"id": FAST_TTFT_ID}, "litellm_params": {}},
    {"model_info": {"id": SLOW_TTFT_ID}, "litellm_params": {}},
]


def _streaming_kwargs(deployment_id: str, start_time: datetime, ttft_seconds: float):
    return {
        "litellm_params": {
            "metadata": {"model_group": MODEL_GROUP},
            "model_info": {"id": deployment_id},
        },
        "stream": True,
        "completion_start_time": start_time + timedelta(seconds=ttft_seconds),
    }


def _recorded_ttft(cache: DualCache, deployment_id: str):
    cached = cache.get_cache(key=f"{MODEL_GROUP}_map") or {}
    return cached.get(deployment_id, {}).get("time_to_first_token_seconds", [])


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False], ids=["sync", "async"])
async def test_streaming_ttft_ranking_ignores_completion_length(sync_mode: bool):
    """Deployment A: TTFT 1s, 50 completion tokens. Deployment B: TTFT 3s, 500
    completion tokens. Dividing TTFT by completion tokens made B look faster
    (3/500 = 0.006 beats 1/50 = 0.02); actual TTFT must win."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)
    start_time = datetime(2026, 1, 1, 12, 0, 0)
    end_time = start_time + timedelta(seconds=10)

    samples = (
        (FAST_TTFT_ID, 1.0, 50),
        (SLOW_TTFT_ID, 3.0, 500),
    )
    for deployment_id, ttft, completion_tokens in samples:
        kwargs = _streaming_kwargs(deployment_id, start_time, ttft)
        response_obj = _chat_response(completion_tokens=completion_tokens)
        if sync_mode:
            handler.log_success_event(
                response_obj=response_obj, kwargs=kwargs, start_time=start_time, end_time=end_time
            )
        else:
            await handler.async_log_success_event(
                response_obj=response_obj, kwargs=kwargs, start_time=start_time, end_time=end_time
            )

    assert _recorded_ttft(cache, FAST_TTFT_ID) == [pytest.approx(1.0)]
    assert _recorded_ttft(cache, SLOW_TTFT_ID) == [pytest.approx(3.0)]

    request_kwargs = {"stream": True, "metadata": {}}
    if sync_mode:
        picked = handler.get_available_deployments(
            model_group=MODEL_GROUP, healthy_deployments=STREAMING_DEPLOYMENTS, request_kwargs=request_kwargs
        )
    else:
        picked = await handler.async_get_available_deployments(
            model_group=MODEL_GROUP, healthy_deployments=STREAMING_DEPLOYMENTS, request_kwargs=request_kwargs
        )

    assert picked is not None
    assert picked["model_info"]["id"] == FAST_TTFT_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False], ids=["sync", "async"])
async def test_ttft_window_keeps_newest_samples_when_full(sync_mode: bool):
    """Float timestamps, as the SDK passes them. Once max_latency_list_size
    samples exist the oldest TTFT is dropped so the window slides."""
    max_size = 3
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache, routing_args={"max_latency_list_size": max_size})
    start_time = 1_700_000_000.0
    ttfts = (0.1, 0.2, 0.3, 0.4)

    for ttft in ttfts:
        kwargs = {
            "litellm_params": {
                "metadata": {"model_group": MODEL_GROUP},
                "model_info": {"id": FAST_TTFT_ID},
            },
            "stream": True,
            "completion_start_time": start_time + ttft,
        }
        response_obj = _chat_response(completion_tokens=1)
        if sync_mode:
            handler.log_success_event(
                response_obj=response_obj, kwargs=kwargs, start_time=start_time, end_time=start_time + 1.0
            )
        else:
            await handler.async_log_success_event(
                response_obj=response_obj, kwargs=kwargs, start_time=start_time, end_time=start_time + 1.0
            )

    assert _recorded_ttft(cache, FAST_TTFT_ID) == [pytest.approx(ttft) for ttft in ttfts[-max_size:]]


@pytest.mark.asyncio
async def test_streaming_routing_ignores_per_token_ttft_samples_from_older_workers():
    """Workers on the previous release share the Redis map and keep writing
    seconds-per-token under the old "time_to_first_token" key during a rolling
    deploy. Those samples favor SLOW; routing must only read the seconds key."""
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)
    cache.set_cache(
        key=f"{MODEL_GROUP}_map",
        value={
            FAST_TTFT_ID: {"time_to_first_token": [0.02], "time_to_first_token_seconds": [1.0]},
            SLOW_TTFT_ID: {"time_to_first_token": [0.006], "time_to_first_token_seconds": [3.0]},
        },
    )

    picked = await handler.async_get_available_deployments(
        model_group=MODEL_GROUP,
        healthy_deployments=STREAMING_DEPLOYMENTS,
        request_kwargs={"stream": True, "metadata": {}},
    )

    assert picked is not None
    assert picked["model_info"]["id"] == FAST_TTFT_ID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cached_entry",
    [{"latency": []}, {"2026-09-05-15-39": {"tpm": 28, "rpm": 1}}],
    ids=["empty_latency_list", "minute_bucket_only_as_cost_based_routing_writes"],
)
async def test_async_get_available_deployments_treats_missing_samples_as_zero_latency(cached_entry):
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)
    cache.set_cache(
        key="gemini-embedding-001_map",
        value={DEPLOYMENT_ID: cached_entry, "slower": {"latency": [0.5]}},
    )
    healthy_deployments = [
        {"model_info": {"id": DEPLOYMENT_ID}, "litellm_params": {}},
        {"model_info": {"id": "slower"}, "litellm_params": {}},
    ]

    picked = await handler.async_get_available_deployments(
        model_group="gemini-embedding-001",
        healthy_deployments=healthy_deployments,
        request_kwargs={"stream": False, "metadata": {}},
    )

    assert picked is not None
    assert picked["model_info"]["id"] == DEPLOYMENT_ID
