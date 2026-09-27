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
from litellm.caching.redis_request_plan import redis_request_plan_scope
from litellm.router_utils.routing_read_batch import declare_routing_prefetch
from tests.unit.caching.redis_batch_fakes import (
    FakeRedisCache,
    RecordingRedisClient,
    json_value,
)

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


class _PrefixPipeline:
    """A pipeline recorder that answers each MGET key from its minute-less prefix."""

    def __init__(self, values_by_key_prefix: dict, error: BaseException | None = None) -> None:
        self.calls: list[tuple] = []
        self.scripts: set = set()
        self.values = values_by_key_prefix
        self.error = error
        self.execute_count = 0

    def _value(self, key: str):
        hit = self.values.get(key.rsplit(":", 1)[0], self.values.get(key))
        return json_value(hit) if hit is not None else None

    def mget(self, keys: list):
        self.calls.append(("mget", tuple(keys)))
        return self

    def get(self, name: str):
        self.calls.append(("get", name))
        return self

    def set(self, name: str, value: object, ex: object = None, **kwargs: object):
        self.calls.append(("set", name))
        return self

    def incrbyfloat(self, name: str, amount: float):
        self.calls.append(("incrbyfloat", name))
        return self

    def expire(self, name: str, ttl: int):
        self.calls.append(("expire", name))
        return self

    def delete(self, name: str):
        self.calls.append(("delete", name))
        return self

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object):
        self.calls.append(("evalsha", sha))
        return self

    async def execute(self, raise_on_error: bool = True) -> list:
        self.execute_count += 1
        if self.error is not None:
            raise self.error
        out = []
        for call in self.calls:
            if call[0] == "mget":
                out.append([self._value(k) for k in call[1]])
            elif call[0] == "get":
                out.append(self._value(call[1]))
            else:
                out.append(True)
        return out


class _PipelineClient:
    def __init__(self, pipe: _PrefixPipeline) -> None:
        self.pipe = pipe

    def pipeline(self, transaction: bool = True) -> _PrefixPipeline:
        return self.pipe


class _PrefetchRedis(FakeRedisCache):
    """A plan-capable Redis double that also spies on the direct shared-read fallback."""

    def __init__(self, values_by_key_prefix: dict, error: BaseException | None = None) -> None:
        self.values = values_by_key_prefix
        self.pipe = _PrefixPipeline(values_by_key_prefix, error)
        super().__init__(RecordingRedisClient([self.pipe]))
        self.fallback_mgets: list[list[str]] = []

    def init_async_client(self) -> _PipelineClient:
        return _PipelineClient(self.pipe)

    async def async_batch_get_cache(self, key_list, parent_otel_span=None):
        self.fallback_mgets.append(list(key_list))
        return {key: self.values.get(key.rsplit(":", 1)[0], self.values.get(key)) for key in key_list}


@pytest.mark.asyncio
async def test_a_declared_prefetch_serves_the_routing_read_without_a_second_redis_trip():
    redis = _PrefetchRedis({"dep-b:anthropic/claude-x:tpm": 5, "dep-a:anthropic/claude-x:tpm": 50})
    router = _router(redis, "usage-based-routing-v2")

    with redis_request_plan_scope() as plan:
        await declare_routing_prefetch(router, _MODEL_GROUP)
        deployment = await router.async_get_available_deployment(
            model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
        )

    assert deployment["model_info"]["id"] == "dep-b"
    assert plan.rounds == 1
    assert redis.pipe.execute_count == 1
    assert [call[0] for call in redis.pipe.calls] == ["mget"]
    families = sorted(
        key.rsplit(":", 1)[0] if ":tpm:" in key or ":rpm:" in key else key
        for key in redis.pipe.calls[0][1]
    )
    assert families == [
        "dep-a:anthropic/claude-x:rpm",
        "dep-a:anthropic/claude-x:tpm",
        "dep-b:anthropic/claude-x:rpm",
        "dep-b:anthropic/claude-x:tpm",
        "deployment:dep-a:cooldown",
        "deployment:dep-b:cooldown",
    ]
    assert redis.fallback_mgets == [], "the shared read was served by the prefetched MGET"


@pytest.mark.asyncio
async def test_a_prefetch_with_no_active_plan_or_matching_batch_is_a_no_op():
    redis = _PrefetchRedis({})
    router = _router(redis, "usage-based-routing-v2")

    await declare_routing_prefetch(router, _MODEL_GROUP)
    assert redis.pipe.execute_count == 0

    with redis_request_plan_scope():
        shuffle = _router(redis, "simple-shuffle")
        await declare_routing_prefetch(shuffle, _MODEL_GROUP)
    assert redis.pipe.execute_count == 0


@pytest.mark.asyncio
async def test_a_failed_prefetch_degrades_like_a_failed_shared_read():
    redis = _PrefetchRedis({}, error=ConnectionError("redis unavailable"))
    router = _router(redis, "usage-based-routing-v2")

    with redis_request_plan_scope():
        await declare_routing_prefetch(router, _MODEL_GROUP)
        with pytest.raises(litellm.RateLimitError, match="No deployments available"):
            await router.async_get_available_deployment(
                model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
            )

    assert redis.pipe.execute_count == 1
    assert len(router.cache.last_redis_batch_access_time) == 0
    assert len(router.cooldown_cache.cooldown_store.last_redis_batch_access_time) == 0


@pytest.mark.asyncio
async def test_a_prefetch_for_a_different_deployment_set_falls_back_to_the_shared_read():
    cooled = _cooldown(seconds=60)
    redis = _PrefetchRedis({"deployment:dep-x:cooldown": cooled})
    other = Router(
        model_list=[_deployment("dep-x")],
        routing_strategy="usage-based-routing-v2",
    )
    other._update_redis_cache(cache=redis)
    router = _router(redis, "usage-based-routing-v2")

    with redis_request_plan_scope() as plan:
        await declare_routing_prefetch(other, _MODEL_GROUP)
        deployment = await router.async_get_available_deployment(
            model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
        )
        await plan.flush()

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert redis.fallback_mgets != [], "cooldown keys differ, so the shared read must run its own MGET"
    assert redis.pipe.execute_count == 1, "the orphaned prefetch is resolved once, not dropped"
    assert (
        other.cooldown_cache.cooldown_store.in_memory_cache.get_cache("deployment:dep-x:cooldown") is not None
    ), "the mismatched prefetch's outcome is applied so its auth-time reservations settle"
