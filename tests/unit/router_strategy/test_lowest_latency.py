#### What this tests ####
#    Latency values recorded by lowest-latency routing must be JSON
#    serializable for non-chat responses too (embeddings/speech/image skip
#    the ModelResponse branch, so the raw timedelta used to leak into the
#    latency list and break the Redis cache sync). Issue #33169.

import copy
import json
import sys
from datetime import datetime, timedelta
from typing import Final
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import litellm
from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.router import Router
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler, RoutingArgs

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


def _embedding_response() -> litellm.EmbeddingResponse:
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
    assert all(not isinstance(value, timedelta) for value in latencies), (
        f"raw timedelta leaked into latency list: {latencies}"
    )
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
    assert all(not isinstance(value, timedelta) for value in latencies), (
        f"raw timedelta leaked into latency list: {latencies}"
    )
    assert latencies[-1] == pytest.approx(3.0)
    json.dumps({"latency": latencies})


def _chat_response(completion_tokens: int) -> litellm.ModelResponse:
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
@pytest.mark.parametrize("sync_mode", [True, False], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("ttft_percentile", "first_samples", "second_samples", "expected_id"),
    [
        (None, [0.1, 0.1, 1.0], [0.3, 0.3, 0.3], SLOW_TTFT_ID),
        (0.5, [0.1, 0.1, 1.0], [0.3, 0.3, 0.3], FAST_TTFT_ID),
        (0.9, [0.1, 0.1, 0.1, 0.1, 1.5], [0.3, 0.3, 0.3, 0.3, 0.3], SLOW_TTFT_ID),
    ],
    ids=["default_average", "p50", "p90"],
)
async def test_streaming_ttft_ranking_percentile(
    sync_mode: bool,
    ttft_percentile: float | None,
    first_samples: list[float],
    second_samples: list[float],
    expected_id: str,
):
    cache = DualCache()
    routing_args = {} if ttft_percentile is None else {"ttft_percentile": ttft_percentile}
    handler = LowestLatencyLoggingHandler(router_cache=cache, routing_args=routing_args)
    cache.set_cache(
        key=f"{MODEL_GROUP}_map",
        value={
            FAST_TTFT_ID: {"time_to_first_token_seconds": first_samples},
            SLOW_TTFT_ID: {"time_to_first_token_seconds": second_samples},
        },
    )

    if sync_mode:
        picked = handler.get_available_deployments(
            model_group=MODEL_GROUP,
            healthy_deployments=STREAMING_DEPLOYMENTS,
            request_kwargs={"stream": True, "metadata": {}},
        )
    else:
        picked = await handler.async_get_available_deployments(
            model_group=MODEL_GROUP,
            healthy_deployments=STREAMING_DEPLOYMENTS,
            request_kwargs={"stream": True, "metadata": {}},
        )

    assert picked is not None
    assert picked["model_info"]["id"] == expected_id


@pytest.mark.parametrize("ttft_percentile", [0, -0.1, 1.1])
def test_ttft_percentile_validation(ttft_percentile: float):
    with pytest.raises(ValidationError):
        RoutingArgs(ttft_percentile=ttft_percentile)


@pytest.mark.parametrize("ttft_percentile", [0.5, 0.9, 0.95, 1.0])
def test_ttft_percentile_accepts_valid_values(ttft_percentile: float):
    assert RoutingArgs(ttft_percentile=ttft_percentile).ttft_percentile == ttft_percentile


@pytest.mark.asyncio
async def test_ttft_percentile_does_not_change_non_streaming_routing():
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache, routing_args={"ttft_percentile": 0.9})
    cache.set_cache(
        key=f"{MODEL_GROUP}_map",
        value={
            FAST_TTFT_ID: {"latency": [1.0], "time_to_first_token_seconds": [0.1]},
            SLOW_TTFT_ID: {"latency": [0.2], "time_to_first_token_seconds": [1.5]},
        },
    )

    picked = await handler.async_get_available_deployments(
        model_group=MODEL_GROUP,
        healthy_deployments=STREAMING_DEPLOYMENTS,
        request_kwargs={"stream": False, "metadata": {}},
    )

    assert picked is not None
    assert picked["model_info"]["id"] == SLOW_TTFT_ID


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


def _latency_router(routing_strategy_args: dict) -> Router:
    return Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": f"openai/{MODEL_GROUP}", "api_key": "sk-fake"},
                "model_info": {"id": deployment_id},
            }
            for deployment_id in (FAST_TTFT_ID, SLOW_TTFT_ID)
        ],
        routing_strategy="latency-based-routing",
        routing_strategy_args=routing_strategy_args,
    )


def _seed_streaming_ttft(router: Router) -> None:
    router.cache.set_cache(
        key=f"{MODEL_GROUP}_map",
        value={
            FAST_TTFT_ID: {"time_to_first_token_seconds": [0.1, 0.1, 1.0]},
            SLOW_TTFT_ID: {"time_to_first_token_seconds": [0.3, 0.3, 0.3]},
        },
    )


async def _pick_streaming(router: Router) -> str:
    picked = await router.async_get_available_deployment(
        model=MODEL_GROUP,
        request_kwargs={"stream": True, "metadata": {}},
    )
    return picked["model_info"]["id"]


@pytest.mark.asyncio
async def test_runtime_routing_strategy_args_update_applies_ttft_percentile():
    """A config reload that adds ttft_percentile must reach the live selector,
    not sit unused until the proxy restarts."""
    router = _latency_router({"max_latency_list_size": 50})
    _seed_streaming_ttft(router)

    assert await _pick_streaming(router) == SLOW_TTFT_ID

    router.update_settings(routing_strategy_args={"max_latency_list_size": 50, "ttft_percentile": 0.5})

    assert await _pick_streaming(router) == FAST_TTFT_ID


@pytest.mark.asyncio
async def test_runtime_routing_strategy_args_update_keeps_previous_args_when_invalid():
    router = _latency_router({"ttft_percentile": 0.5})
    _seed_streaming_ttft(router)

    router.update_settings(routing_strategy_args={"ttft_percentile": 5})

    assert await _pick_streaming(router) == FAST_TTFT_ID


@pytest.mark.asyncio
async def test_runtime_routing_strategy_args_update_is_a_noop_without_a_selector():
    """simple-shuffle has no selector to re-link, so an args update must leave
    the router alone instead of blowing up on a missing selector attribute."""
    router = Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": f"openai/{MODEL_GROUP}", "api_key": "sk-fake"},
                "model_info": {"id": FAST_TTFT_ID},
            }
        ],
        routing_strategy="simple-shuffle",
    )

    router.update_settings(routing_strategy_args={"ttl": 5})

    assert router.routing_strategy_args == {"ttl": 5}
    assert await _pick_streaming(router) == FAST_TTFT_ID


FROZEN_NOW = datetime(2026, 1, 15, 12, 30, 15)
FROZEN_EPOCH = FROZEN_NOW.timestamp()


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW if tz is None else FROZEN_NOW.astimezone(tz)


@pytest.fixture
def frozen_latency_clock(monkeypatch):
    monkeypatch.setattr("litellm.router_strategy.lowest_latency.datetime", _FrozenDatetime)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@pytest.mark.usefixtures("frozen_latency_clock")
async def test_latency_memory_leak(sync_mode):
    """
    Test to make sure there's no memory leak caused by lowest latency routing

    - make 10 calls -> check memory
    - make 11th call -> no change in memory
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(router_cache=test_cache)
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = FROZEN_EPOCH
    response_obj = {"usage": {"total_tokens": 50}}
    end_time = start_time + 5
    for _ in range(10):
        if sync_mode:
            lowest_latency_logger.log_success_event(
                response_obj=response_obj,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            await lowest_latency_logger.async_log_success_event(
                response_obj=response_obj,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
            )
    latency_key = f"{model_group}_map"
    cache_value = copy.deepcopy(test_cache.get_cache(key=latency_key))

    if sync_mode:
        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        await lowest_latency_logger.async_log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    new_cache_value = test_cache.get_cache(key=latency_key)
    assert get_size(new_cache_value) <= get_size(cache_value), (
        f"Memory leak detected in function call! new_cache size={get_size(new_cache_value)}, old cache size={get_size(cache_value)}"
    )


def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


@pytest.mark.usefixtures("frozen_latency_clock")
def test_latency_updated():
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(router_cache=test_cache)
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = FROZEN_EPOCH
    response_obj = {"usage": {"total_tokens": 50}}
    end_time = start_time + 5
    lowest_latency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    latency_key = f"{model_group}_map"
    assert end_time - start_time == test_cache.get_cache(key=latency_key)[deployment_id]["latency"][0]


@pytest.mark.usefixtures("frozen_latency_clock")
def test_get_available_deployments():
    test_cache = DualCache()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/gpt-4.1-mini"},
            "model_info": {"id": "1234"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/gpt-4.1-mini"},
            "model_info": {"id": "5678"},
        },
    ]
    lowest_latency_logger = LowestLatencyLoggingHandler(router_cache=test_cache)
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = FROZEN_EPOCH
    response_obj = {"usage": {"total_tokens": 50}}
    end_time = start_time + 3
    lowest_latency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    deployment_id = "5678"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = FROZEN_EPOCH
    response_obj = {"usage": {"total_tokens": 20}}
    end_time = start_time + 2
    lowest_latency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    print(lowest_latency_logger.get_available_deployments(model_group=model_group, healthy_deployments=model_list))
    assert (
        lowest_latency_logger.get_available_deployments(model_group=model_group, healthy_deployments=model_list)[
            "model_info"
        ]["id"]
        == "5678"
    )


@pytest.mark.usefixtures("frozen_latency_clock")
def test_router_get_available_deployments():
    """
    Test if routers 'get_available_deployments' returns the fastest deployment
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        set_verbose=False,
        num_retries=3,
    )

    deployment_id = 1
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 1},
        }
    }
    start_time = FROZEN_EPOCH
    response_obj = {"usage": {"total_tokens": 50}}
    end_time = start_time + 3
    router.lowestlatency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    deployment_id = 2
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 2},
        }
    }
    start_time = FROZEN_EPOCH
    response_obj = {"usage": {"total_tokens": 20}}
    end_time = start_time + 2
    router.lowestlatency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    print(router.get_available_deployment(model="azure-model"))
    assert router.get_available_deployment(model="azure-model")["model_info"]["id"] == "2"


@pytest.mark.parametrize("buffer", [0, 1])
@pytest.mark.asyncio
@pytest.mark.usefixtures("frozen_latency_clock")
async def test_lowest_latency_routing_buffer(buffer):
    """
    Allow shuffling calls within a certain latency buffer
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        set_verbose=False,
        num_retries=3,
        routing_strategy_args={"lowest_latency_buffer": buffer},
    )

    deployment_id = 1
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 1},
        }
    }
    start_time = FROZEN_EPOCH
    response_obj = {"usage": {"total_tokens": 50}}
    end_time = start_time + 3
    router.lowestlatency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    deployment_id = 2
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 2},
        }
    }
    start_time = FROZEN_EPOCH
    response_obj = {"usage": {"total_tokens": 20}}
    end_time = start_time + 2
    router.lowestlatency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    selected_deployments = {}
    for _ in range(50):
        print(router.get_available_deployment(model="azure-model"))
        selected_deployments[router.get_available_deployment(model="azure-model")["model_info"]["id"]] = 1

    if buffer == 0:
        assert len(selected_deployments.keys()) == 1
    else:
        assert len(selected_deployments.keys()) == 2


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@pytest.mark.usefixtures("frozen_latency_clock")
async def test_lowest_latency_routing_time_to_first_token(sync_mode):
    """
    If a deployment has
    - a fast time to first token
    - slow latency/output token

    test if:
    - for streaming, the deployment with fastest time to first token is picked
    - for non-streaming, fastest overall deployment is picked
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        set_verbose=False,
        num_retries=3,
    )
    deployment_id = 1
    start_time = FROZEN_NOW
    one_second_later = start_time + timedelta(seconds=1)

    three_seconds_later = start_time + timedelta(seconds=3)
    four_seconds_later = start_time + timedelta(seconds=4)

    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 1},
        },
        "stream": True,
        "completion_start_time": one_second_later,
    }

    response_obj = litellm.ModelResponse(usage=litellm.Usage(completion_tokens=50, total_tokens=50))
    end_time = four_seconds_later

    if sync_mode:
        router.lowestlatency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        await router.lowestlatency_logger.async_log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    deployment_id = 2
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 2},
        },
        "stream": True,
        "completion_start_time": three_seconds_later,
    }
    response_obj = litellm.ModelResponse(usage=litellm.Usage(completion_tokens=50, total_tokens=50))
    end_time = three_seconds_later
    if sync_mode:
        router.lowestlatency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        await router.lowestlatency_logger.async_log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )

    """
    TESTING

    - expect deployment 1 to be picked for streaming
    - expect deployment 2 to be picked for non-streaming
    """
    selected_deployments = {}
    for _ in range(3):
        print(router.get_available_deployment(model="azure-model"))
        selected_deployments[router.get_available_deployment(model="azure-model")["model_info"]["id"]] = 1

    assert len(selected_deployments.keys()) == 1
    assert "2" in list(selected_deployments.keys())

    selected_deployments = {}
    for _ in range(50):
        print(router.get_available_deployment(model="azure-model"))
        selected_deployments[
            router.get_available_deployment(model="azure-model", request_kwargs={"stream": True})["model_info"]["id"]
        ] = 1

    assert len(selected_deployments.keys()) == 1
    assert "1" in list(selected_deployments.keys())


@pytest.mark.usefixtures("frozen_latency_clock")
def test_latency_list_trimming_discards_oldest_entry():
    """
    When the latency list reaches max_latency_list_size, the oldest entry is
    discarded to make room for new entries. The newest entry is appended at
    the end of the list.
    """
    max_size = 3
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, routing_args={"max_latency_list_size": max_size}
    )

    model_group = "gpt-3.5-turbo"
    deployment_id = "test-deployment"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": deployment_id},
        }
    }

    latencies_to_add = []
    for i in range(max_size + 1):
        start_time = FROZEN_EPOCH
        response_obj = {"usage": {"total_tokens": 1, "completion_tokens": 1}}
        expected_latency = float(i + 1)
        end_time = start_time + expected_latency
        latencies_to_add.append(expected_latency)

        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )

    latency_key = f"{model_group}_map"
    cached_data = test_cache.get_cache(key=latency_key)
    latency_list = cached_data[deployment_id]["latency"]

    assert len(latency_list) == max_size, f"Expected {max_size} entries, got {len(latency_list)}"

    newest_latency = latencies_to_add[-1]
    oldest_latency = latencies_to_add[0]
    tolerance = 0.1

    assert abs(latency_list[-1] - newest_latency) < tolerance, (
        f"Newest latency {newest_latency} should be at end, got {latency_list[-1]}"
    )

    for latency in latency_list:
        assert abs(latency - oldest_latency) > tolerance, (
            f"Oldest latency {oldest_latency} should have been discarded, found {latency}"
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("frozen_latency_clock")
async def test_latency_list_trimming_discards_oldest_entry_async():
    """
    Async counterpart: the oldest entry is discarded when the latency list is
    trimmed.
    """
    max_size = 3
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, routing_args={"max_latency_list_size": max_size}
    )

    model_group = "gpt-3.5-turbo"
    deployment_id = "test-deployment"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": deployment_id},
        }
    }

    latencies_to_add = []
    for i in range(max_size + 1):
        start_time = FROZEN_EPOCH
        response_obj = {"usage": {"total_tokens": 1, "completion_tokens": 1}}
        expected_latency = float(i + 1)
        end_time = start_time + expected_latency
        latencies_to_add.append(expected_latency)

        await lowest_latency_logger.async_log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )

    latency_key = f"{model_group}_map"
    cached_data = await test_cache.async_get_cache(key=latency_key)
    latency_list = cached_data[deployment_id]["latency"]

    assert len(latency_list) == max_size

    newest_latency = latencies_to_add[-1]
    oldest_latency = latencies_to_add[0]
    tolerance = 0.1

    assert abs(latency_list[-1] - newest_latency) < tolerance, (
        f"Newest latency {newest_latency} should be at end of list"
    )

    for latency in latency_list:
        assert abs(latency - oldest_latency) > tolerance, f"Oldest latency {oldest_latency} should have been discarded"


@pytest.mark.asyncio
@pytest.mark.usefixtures("frozen_latency_clock")
async def test_timeout_penalty_discards_oldest_entry():
    """
    Timeout penalties (1000.0) are appended to the latency list and, when the
    list is full, the oldest entry is discarded.
    """
    max_size = 3
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, routing_args={"max_latency_list_size": max_size}
    )

    model_group = "gpt-3.5-turbo"
    deployment_id = "test-deployment"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": deployment_id},
        }
    }

    for i in range(max_size):
        start_time = FROZEN_EPOCH
        response_obj = {"usage": {"total_tokens": 1, "completion_tokens": 1}}
        end_time = start_time + float(i + 1)

        await lowest_latency_logger.async_log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )

    timeout_kwargs = {
        **kwargs,
        "exception": litellm.Timeout(message="Request timed out", model="test-model", llm_provider="test"),
    }

    await lowest_latency_logger.async_log_failure_event(
        kwargs=timeout_kwargs,
        response_obj=None,
        start_time=FROZEN_EPOCH,
        end_time=FROZEN_EPOCH + 30,
    )

    latency_key = f"{model_group}_map"
    cached_data = await test_cache.async_get_cache(key=latency_key)
    latency_list = cached_data[deployment_id]["latency"]

    assert len(latency_list) == max_size

    assert latency_list[-1] == 1000.0, f"Timeout penalty should be at end of list, got {latency_list[-1]}"

    tolerance = 0.1
    for latency in latency_list[:-1]:
        assert abs(latency - 1.0) > tolerance, f"Oldest latency 1.0 should have been discarded, found {latency}"


@pytest.mark.usefixtures("frozen_latency_clock")
def test_list_order_preserved_after_multiple_trims():
    """
    After many trims, the list still holds the most recent `max_size` entries
    in insertion order (oldest at index 0, newest at index -1).
    """
    max_size = 3
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, routing_args={"max_latency_list_size": max_size}
    )

    model_group = "gpt-3.5-turbo"
    deployment_id = "test-deployment"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": deployment_id},
        }
    }

    all_latencies = []
    for i in range(10):
        start_time = FROZEN_EPOCH
        response_obj = {"usage": {"total_tokens": 1, "completion_tokens": 1}}
        expected_latency = float(i + 1)
        end_time = start_time + expected_latency
        all_latencies.append(expected_latency)

        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )

    latency_key = f"{model_group}_map"
    cached_data = test_cache.get_cache(key=latency_key)
    latency_list = cached_data[deployment_id]["latency"]

    assert len(latency_list) == max_size

    expected_remaining = all_latencies[-max_size:]
    tolerance = 0.1

    for i, expected in enumerate(expected_remaining):
        assert abs(latency_list[i] - expected) < tolerance, f"At index {i}, expected ~{expected}, got {latency_list[i]}"


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
@pytest.mark.usefixtures("frozen_latency_clock")
async def test_usage_limits_filter_sync_and_async_logging(async_mode: bool) -> None:
    cache: Final = DualCache()
    handler: Final = LowestLatencyLoggingHandler(router_cache=cache)
    model_group: Final = "usage-limits"
    deployments: Final = [
        {
            "model_name": model_group,
            "litellm_params": {"model": "openai/gpt-4o-mini", "rpm": 1},
            "model_info": {"id": "rpm-limited"},
        },
        {
            "model_name": model_group,
            "litellm_params": {"model": "openai/gpt-4o-mini", "tpm": 1},
            "model_info": {"id": "tpm-limited"},
        },
        {
            "model_name": model_group,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "rpm": 3,
                "tpm": 1000,
            },
            "model_info": {"id": "available"},
        },
    ]
    response: Final = _chat_response(completion_tokens=4)

    log_kwargs: Final = tuple(
        {
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": deployment_id},
            }
        }
        for deployment_id in ("rpm-limited", "tpm-limited")
    )
    for kwargs in log_kwargs:
        if async_mode:
            await handler.async_log_success_event(
                kwargs=kwargs,
                response_obj=response,
                start_time=FROZEN_EPOCH,
                end_time=FROZEN_EPOCH + 1,
            )
        else:
            handler.log_success_event(
                kwargs=kwargs,
                response_obj=response,
                start_time=FROZEN_EPOCH,
                end_time=FROZEN_EPOCH + 1,
            )

    selected: Final = (
        await handler.async_get_available_deployments(
            model_group=model_group,
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "check"}],
        )
        if async_mode
        else handler.get_available_deployments(
            model_group=model_group,
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "check"}],
        )
    )
    assert selected is not None
    assert selected["model_info"]["id"] == "available"

    blocked: Final = (
        await handler.async_get_available_deployments(
            model_group=model_group,
            healthy_deployments=deployments[:2],
            messages=[{"role": "user", "content": "check"}],
        )
        if async_mode
        else handler.get_available_deployments(
            model_group=model_group,
            healthy_deployments=deployments[:2],
            messages=[{"role": "user", "content": "check"}],
        )
    )
    assert blocked is None


@pytest.mark.usefixtures("frozen_latency_clock")
@pytest.mark.asyncio
async def test_router_uses_lowest_latency_deployment() -> None:
    model_group: Final = "latency-routing"
    router: Final = Router(
        model_list=[
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "mock_response": _chat_response(completion_tokens=4),
                },
                "model_info": {"id": "slow"},
            },
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "mock_response": _chat_response(completion_tokens=4),
                },
                "model_info": {"id": "fast"},
            },
        ],
        routing_strategy="latency-based-routing",
        num_retries=0,
    )
    router.lowestlatency_logger.log_success_event(
        kwargs={
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": "slow"},
            }
        },
        response_obj={"usage": {"total_tokens": 1}},
        start_time=FROZEN_EPOCH,
        end_time=FROZEN_EPOCH + 10,
    )
    router.lowestlatency_logger.log_success_event(
        kwargs={
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": "fast"},
            }
        },
        response_obj={"usage": {"total_tokens": 1}},
        start_time=FROZEN_EPOCH,
        end_time=FROZEN_EPOCH + 1,
    )

    selected: Final = router.get_available_deployment(model=model_group)
    response: Final = await router.acompletion(
        model=model_group,
        messages=[{"role": "user", "content": "latency check"}],
    )

    assert selected["model_info"]["id"] == "fast"
    assert response._hidden_params["model_id"] == "fast"


@pytest.mark.usefixtures("frozen_latency_clock")
@pytest.mark.asyncio
async def test_router_streaming_uses_lowest_latency_deployment() -> None:
    model_group: Final = "streaming-latency-routing"
    router: Final = Router(
        model_list=[
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "mock_response": "Hello world",
                },
                "model_info": {"id": "slow"},
            },
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "mock_response": "Hello world",
                },
                "model_info": {"id": "fast"},
            },
        ],
        routing_strategy="latency-based-routing",
        num_retries=0,
    )
    router.lowestlatency_logger.log_success_event(
        kwargs={
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": "slow"},
            }
        },
        response_obj={"usage": {"total_tokens": 1}},
        start_time=FROZEN_EPOCH,
        end_time=FROZEN_EPOCH + 10,
    )
    router.lowestlatency_logger.log_success_event(
        kwargs={
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": "fast"},
            }
        },
        response_obj={"usage": {"total_tokens": 1}},
        start_time=FROZEN_EPOCH,
        end_time=FROZEN_EPOCH + 1,
    )
    selected: Final = router.get_available_deployment(model=model_group)
    response: Final = await router.acompletion(
        model=model_group,
        messages=[{"role": "user", "content": "streaming latency check"}],
        stream=True,
    )
    chunks: Final = tuple([chunk async for chunk in response])

    assert selected["model_info"]["id"] == "fast"
    assert response._hidden_params["model_id"] == "fast"
    assert "".join(
        chunk.choices[0].delta.content or "" for chunk in chunks
    ) == "Hello world"


def test_latency_cache_honors_custom_ttl() -> None:
    clock: Final = Mock(return_value=100.0)
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock))
    handler: Final = LowestLatencyLoggingHandler(
        router_cache=cache, routing_args={"ttl": 3}
    )

    handler.log_success_event(
        kwargs={
            "litellm_params": {
                "metadata": {"model_group": "ttl-group"},
                "model_info": {"id": "deployment"},
            }
        },
        response_obj={"usage": {"total_tokens": 1}},
        start_time=100.0,
        end_time=101.0,
    )

    latency_key: Final = "ttl-group_map"
    cached: Final = cache.get_cache(key=latency_key)
    assert isinstance(cached, dict)
    assert cached["deployment"]["latency"] == [1.0]
    assert cache.in_memory_cache.ttl_dict[latency_key] == 103.0

    clock.return_value = 104.0
    assert cache.get_cache(key=latency_key) is None


@pytest.mark.usefixtures("frozen_latency_clock")
@pytest.mark.asyncio
async def test_router_model_group_usage_increases_by_logged_tokens() -> None:
    model_group: Final = "usage-group"
    router: Final = Router(
        model_list=[
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "tpm": 100,
                    "mock_response": _chat_response(completion_tokens=5),
                },
                "model_info": {"id": "usage-deployment"},
            }
        ],
        num_retries=0,
    )

    await router.acompletion(
        model=model_group,
        messages=[{"role": "user", "content": "usage check"}],
    )
    initial_usage: Final = await router.get_model_group_usage(model_group=model_group)
    await router.acompletion(
        model=model_group,
        messages=[{"role": "user", "content": "usage check"}],
    )
    updated_usage: Final = await router.get_model_group_usage(model_group=model_group)

    assert updated_usage[0] == initial_usage[0] + 15
