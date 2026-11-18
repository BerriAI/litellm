import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pytest

from litellm.redis.stores.connection import connect
from litellm.redis.stores.deployment_health import DeploymentHealthStates, DeploymentHealthStore
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, RouterCooldownStore
from litellm.redis.stores.router_usage import RouterUsageStore
from litellm.router_rust.routing_stores import (
    CACHE_STEP,
    AddRequestStep,
    HealthStates,
    HealthStatesStep,
    IncrementAllowedFailsStep,
    PutCooldownStep,
    RoutingStores,
    UsageStep,
)
from litellm.router_utils.cooldown_cache import CooldownCache
from litellm.router_utils.health_state_cache import DeploymentHealthCache
from tests.unit.redis.stores.test_router_cooldowns import FakeCooldowns

_ENTRY: Final[CooldownCacheValue] = {
    "exception_received": "RateLimitError",
    "status_code": "429",
    "timestamp": 1000.0,
    "cooldown_time": 60,
}


@dataclass
class SlowRedis(FakeCooldowns):
    """A cooldown handle whose every Redis write takes a round trip, noting whether another request on the same
    event loop ran while it waited."""

    other_request: asyncio.Event | None = None
    overlapped: bool | None = None

    def _waited(self) -> None:
        assert self.other_request is not None
        self.overlapped = self.other_request.is_set()

    def put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> None:
        self._waited()
        super().put(model_id, cooldown, ttl_seconds)

    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> int:
        self._waited()
        return super().increment_allowed_fails(model_id, suffix, ttl_seconds)

    async def async_put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> None:
        await asyncio.sleep(0.01)
        self.put(model_id, cooldown, ttl_seconds)

    async def async_increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> int:
        await asyncio.sleep(0.01)
        return self.increment_allowed_fails(model_id, suffix, ttl_seconds)


class NoHealthStates:
    async def async_get_states(self) -> DeploymentHealthStates | None:
        return None

    def get_states(self) -> DeploymentHealthStates | None:
        return None


def _health_cache(written: Mapping[str, object] | None) -> DeploymentHealthCache:
    """The native health store holding what a writer left, decoded the way it is when read back from Redis."""
    handle: Final = connect(None).deployment_health()
    if written is not None:
        handle.set_sync(written, 90.0)
    return DeploymentHealthCache(store=DeploymentHealthStore(handle), staleness_threshold=60.0)


def _stores(health: HealthStates) -> RoutingStores:
    cooldowns: Final = CooldownCache(store=RouterCooldownStore(FakeCooldowns()), default_cooldown_time=60)
    return RoutingStores(cooldowns, RouterUsageStore(connect(None).router_usage()), health)


async def _another_request(ran: asyncio.Event) -> None:
    ran.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "answer"),
    [
        ({"op": "put_cooldown", "id": "dep", "cooldown": _ENTRY, "ttl": 60.0}, None),
        ({"op": "increment_allowed_fails", "id": "dep", "suffix": "timeout", "ttl": 60.0}, 1),
    ],
)
async def test_an_async_cooldown_step_waiting_on_redis_lets_other_requests_run(
    raw: Mapping[str, object], answer: object
) -> None:
    """router-core's failure path writes cooldowns and allowed-fails counts; on an async call that Redis round trip
    must not hold the event loop, or every request on the worker stalls behind it."""
    other_request: Final = asyncio.Event()
    handle: Final = SlowRedis(has_redis=True, other_request=other_request)
    cooldowns: Final = CooldownCache(store=RouterCooldownStore(handle), default_cooldown_time=60)
    stores: Final = RoutingStores(cooldowns, RouterUsageStore(connect(None).router_usage()), NoHealthStates())
    step: Final = CACHE_STEP.validate_python(raw)
    assert isinstance(step, PutCooldownStep | IncrementAllowedFailsStep)

    answering: Final = asyncio.create_task(stores.answer(step))
    other: Final = asyncio.create_task(_another_request(other_request))
    handled: Final = await answering
    await other

    assert handled == answer
    assert handle.overlapped is True
    assert handle.targets == ["router_cooldowns"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "states"),
    [
        pytest.param(
            {"dep": {"is_healthy": False, "timestamp": 95.0, "reason": "timeout"}},
            {"dep": {"is_healthy": False, "timestamp": 95.0}},
            id="unhealthy",
        ),
        pytest.param({"dep": {"timestamp": 95.0}}, {"dep": {"is_healthy": True, "timestamp": 95.0}}, id="no_flag"),
        pytest.param(
            {"dep": {"is_healthy": 0, "timestamp": 95}},
            {"dep": {"is_healthy": False, "timestamp": 95.0}},
            id="falsy_flag",
        ),
        pytest.param(
            {"dep": {"is_healthy": False, "timestamp": "soon"}},
            {"dep": {"is_healthy": False, "timestamp": 0.0}},
            id="timestamp_that_is_no_number",
        ),
        pytest.param({"dep": 1}, {}, id="not_a_record"),
        pytest.param(None, {}, id="nothing_cached"),
    ],
)
async def test_health_states_reach_router_core_as_mains_python_filter_read_them(
    raw: Mapping[str, object] | None, states: Mapping[str, object]
) -> None:
    """router-core takes each state as a strict bool and float, so the store's decoder applies main's leniency
    (a falsy flag is unhealthy, a missing one healthy, a bad timestamp stale) before a record crosses."""
    stores: Final = _stores(_health_cache(raw))
    step: Final = CACHE_STEP.validate_python({"op": "health_states"})
    assert isinstance(step, HealthStatesStep)

    assert await stores.answer(step) == states
    assert stores.answer_sync(step) == states


@pytest.mark.asyncio
async def test_a_failed_request_counts_one_request_and_no_tokens() -> None:
    stores: Final = _stores(NoHealthStates())
    window: Final = {"deployment_id": "dep", "model": "gpt", "minute": "09-05"}
    counted: Final = CACHE_STEP.validate_python({"op": "add_request", "window": window})
    read: Final = CACHE_STEP.validate_python({"op": "usage", "window": window})
    assert isinstance(counted, AddRequestStep)
    assert isinstance(read, UsageStep)

    await stores.answer(counted)
    stores.answer_sync(counted)

    assert await stores.answer(read) == (None, 2)
