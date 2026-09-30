"""
One Redis round trip per request for the router's pre-call reads.

Before `RoutingReadBatch`, `async_get_available_deployment` issued one MGET for the cooldown keys
(`CooldownCache`) and a second one for the tpm/rpm counters (`LowestTPMLoggingHandler_v2`).
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm import Router
from litellm.caching.redis_cache import RedisCache

_MODEL_GROUP = "claude"
_MESSAGES = [{"role": "user", "content": "ping"}]


def _deployment(deployment_id: str) -> dict:
    return {
        "model_name": _MODEL_GROUP,
        "litellm_params": {"model": "anthropic/claude-x", "api_key": "test", "mock_response": "pong"},
        "model_info": {"id": deployment_id},
    }


def _redis_answering(values_by_key_prefix: dict[str, object]) -> MagicMock:
    """A Redis double that answers each key from its minute-less prefix and records every MGET."""

    def _mget(key_list, parent_otel_span=None):
        return {key: values_by_key_prefix.get(key.rsplit(":", 1)[0], values_by_key_prefix.get(key)) for key in key_list}

    redis = MagicMock(spec=RedisCache)
    redis.async_batch_get_cache = AsyncMock(side_effect=_mget)
    return redis


def _router(redis: MagicMock, routing_strategy: str) -> Router:
    router = Router(
        model_list=[_deployment("dep-a"), _deployment("dep-b")],
        routing_strategy=routing_strategy,
    )
    router._update_redis_cache(cache=redis)
    return router


def _redis_key_families(redis: MagicMock) -> list[list[str]]:
    return [
        sorted(key.rsplit(":", 1)[0] if ":tpm:" in key or ":rpm:" in key else key for key in call.args[0])
        for call in redis.async_batch_get_cache.await_args_list
    ]


def _cooldown(seconds: float) -> dict:
    return {"exception_received": "429", "status_code": "429", "timestamp": time.time(), "cooldown_time": seconds}


@pytest.mark.asyncio
async def test_usage_based_routing_reads_cooldowns_and_counters_in_one_redis_round_trip():
    redis = _redis_answering({})
    router = _router(redis, "usage-based-routing-v2")

    deployment = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert _redis_key_families(redis) == [
        [
            "dep-a:anthropic/claude-x:rpm",
            "dep-a:anthropic/claude-x:tpm",
            "dep-b:anthropic/claude-x:rpm",
            "dep-b:anthropic/claude-x:tpm",
            "deployment:dep-a:cooldown",
            "deployment:dep-b:cooldown",
        ]
    ], "cooldown state and usage counters must arrive in one MGET"


@pytest.mark.asyncio
async def test_simple_shuffle_still_reads_only_cooldowns():
    redis = _redis_answering({})
    router = _router(redis, "simple-shuffle")

    await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert _redis_key_families(redis) == [["deployment:dep-a:cooldown", "deployment:dep-b:cooldown"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tpm_a", "tpm_b", "expected"),
    [(100, 10, "dep-b"), (10, 100, "dep-a"), (None, 10, "dep-a"), (10, None, "dep-b")],
)
async def test_batched_counters_pick_the_deployment_the_strategy_picks_reading_alone(tpm_a, tpm_b, expected):
    counters = {"dep-a:anthropic/claude-x:tpm": tpm_a, "dep-b:anthropic/claude-x:tpm": tpm_b}
    routed = _router(_redis_answering(counters), "usage-based-routing-v2")
    alone = _router(_redis_answering(counters), "usage-based-routing-v2")

    routed_choice = await routed.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )
    alone_choice = await alone.lowesttpm_logger_v2.async_get_available_deployments(
        model_group=_MODEL_GROUP, healthy_deployments=alone.model_list, messages=_MESSAGES
    )

    assert routed_choice["model_info"]["id"] == alone_choice["model_info"]["id"] == expected


@pytest.mark.asyncio
async def test_batched_read_still_excludes_a_cooled_down_deployment():
    redis = _redis_answering(
        {
            "dep-a:anthropic/claude-x:tpm": 100,
            "dep-b:anthropic/claude-x:tpm": 10,
            "deployment:dep-b:cooldown": _cooldown(seconds=60),
        }
    )
    router = _router(redis, "usage-based-routing-v2")

    deployment = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] == "dep-a", "dep-b has the lowest tpm but is cooling down"
    assert redis.async_batch_get_cache.await_count == 1


@pytest.mark.asyncio
async def test_batched_read_ignores_an_expired_cooldown():
    redis = _redis_answering(
        {
            "dep-a:anthropic/claude-x:tpm": 100,
            "dep-b:anthropic/claude-x:tpm": 10,
            "deployment:dep-b:cooldown": _cooldown(seconds=-1),
        }
    )
    router = _router(redis, "usage-based-routing-v2")

    deployment = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] == "dep-b"


@pytest.mark.asyncio
async def test_a_failed_batched_read_degrades_like_the_two_failed_reads_did():
    redis = MagicMock(spec=RedisCache)
    redis.async_batch_get_cache = AsyncMock(side_effect=ConnectionError("redis unavailable"))
    routed = _router(redis, "usage-based-routing-v2")
    alone = _router(redis, "usage-based-routing-v2")

    with pytest.raises(litellm.RateLimitError, match="No deployments available") as routed_error:
        await routed.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)
    with pytest.raises(litellm.RateLimitError, match="No deployments available") as alone_error:
        await alone.lowesttpm_logger_v2.async_get_available_deployments(
            model_group=_MODEL_GROUP, healthy_deployments=alone.model_list, messages=_MESSAGES
        )

    assert str(routed_error.value) == str(alone_error.value)
    assert len(routed.cache.last_redis_batch_access_time) == 0, "a failed read must not throttle the next one"
    assert len(routed.cooldown_cache.cooldown_store.last_redis_batch_access_time) == 0


@pytest.mark.asyncio
async def test_a_failed_batched_read_leaves_simple_shuffle_routing():
    redis = MagicMock(spec=RedisCache)
    redis.async_batch_get_cache = AsyncMock(side_effect=ConnectionError("redis unavailable"))
    router = _router(redis, "simple-shuffle")

    deployment = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
