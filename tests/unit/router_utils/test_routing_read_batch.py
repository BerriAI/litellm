"""
The router's pre-call reads. Cooldowns go through the native store, and the tpm/rpm selector
counters (`LowestTPMLoggingHandler_v2`) ride one MGET on the Python Redis.
"""

import time
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm import Router
from litellm.caching.redis_cache import RedisCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from tests.unit.conftest import redis_cache_double

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

    redis = redis_cache_double()
    redis.async_batch_get_cache = AsyncMock(side_effect=_mget)
    return redis


def _router(redis: MagicMock, routing_strategy: str) -> Router:
    router = Router(
        model_list=[_deployment("dep-a"), _deployment("dep-b")],
        routing_strategy=routing_strategy,
    )
    router.update_redis_cache(cache=redis)
    return router


def _redis_key_families(redis: MagicMock) -> list[list[str]]:
    return [
        sorted(key.rsplit(":", 1)[0] if ":tpm:" in key or ":rpm:" in key else key for key in call.args[0])
        for call in redis.async_batch_get_cache.await_args_list
    ]


def _cool_down(router: Router, model_id: str, started: float) -> None:
    router.cooldown_cache.cooldown_store.put(
        model_id, {"exception_received": "429", "status_code": "429", "timestamp": started, "cooldown_time": 60}, 60
    )


@pytest.mark.asyncio
async def test_usage_based_routing_reads_the_counters_in_one_python_redis_round_trip():
    redis = _redis_answering({})
    router = _router(redis, "usage-based-routing-v2")

    deployment = await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert _redis_key_families(redis) == [
        [
            "dep-a:anthropic/claude-x:rpm",
            "dep-a:anthropic/claude-x:tpm",
            "dep-b:anthropic/claude-x:rpm",
            "dep-b:anthropic/claude-x:tpm",
        ]
    ], "the usage counters must arrive in one MGET, without cooldown keys"


@pytest.mark.asyncio
async def test_usage_based_routing_still_batches_when_the_strategy_is_a_fixed_signature_subclass():
    class OldSignatureV2(LowestTPMLoggingHandler_v2):
        async def async_get_available_deployments(
            self,
            model_group: str,
            healthy_deployments: list,
            messages: list[dict[str, str]] | None = None,
            input: str | list | None = None,
        ):
            return await super().async_get_available_deployments(
                model_group=model_group,
                healthy_deployments=healthy_deployments,
                messages=messages,
                input=input,
            )

    redis: Final = _redis_answering({})
    router: Final = _router(redis, "usage-based-routing-v2")
    router.lowesttpm_logger_v2 = OldSignatureV2(router_cache=router.cache)
    router.cache.async_batch_get_cache = AsyncMock(wraps=router.cache.async_batch_get_cache)

    deployment: Final = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert _redis_key_families(redis) == [
        [
            "dep-a:anthropic/claude-x:rpm",
            "dep-a:anthropic/claude-x:tpm",
            "dep-b:anthropic/claude-x:rpm",
            "dep-b:anthropic/claude-x:tpm",
        ]
    ], "the subclassed strategy must still get the batched read, not a second MGET"
    router.cache.async_batch_get_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_simple_shuffle_reads_nothing_from_the_python_redis():
    redis = _redis_answering({})
    router = _router(redis, "simple-shuffle")

    await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert _redis_key_families(redis) == []


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
    redis = _redis_answering({"dep-a:anthropic/claude-x:tpm": 100, "dep-b:anthropic/claude-x:tpm": 10})
    router = _router(redis, "usage-based-routing-v2")
    _cool_down(router, "dep-b", started=time.time())

    deployment = await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert deployment["model_info"]["id"] == "dep-a", "dep-b has the lowest tpm but is cooling down"
    assert redis.async_batch_get_cache.await_count == 1


@pytest.mark.asyncio
async def test_batched_read_ignores_an_expired_cooldown():
    redis = _redis_answering({"dep-a:anthropic/claude-x:tpm": 100, "dep-b:anthropic/claude-x:tpm": 10})
    router = _router(redis, "usage-based-routing-v2")
    _cool_down(router, "dep-b", started=time.time() - 61)

    deployment = await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert deployment["model_info"]["id"] == "dep-b"


@pytest.mark.asyncio
async def test_a_failed_batched_read_degrades_like_the_two_failed_reads_did():
    redis = redis_cache_double()
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


@pytest.mark.asyncio
async def test_a_failed_batched_read_leaves_simple_shuffle_routing():
    redis = redis_cache_double()
    redis.async_batch_get_cache = AsyncMock(side_effect=ConnectionError("redis unavailable"))
    router = _router(redis, "simple-shuffle")

    deployment = await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
