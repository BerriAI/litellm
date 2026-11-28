import importlib
import inspect
import pkgutil
from collections.abc import Awaitable, Callable, Mapping
from types import FunctionType
from typing import Final

import pytest

import litellm.redis.stores
from litellm.redis.stores.active_teams import ActiveTeamStore
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore, AuthObject
from litellm.redis.stores.cache_invalidation import CacheInvalidation
from litellm.redis.stores.connection import connect
from litellm.redis.stores.deployment_health import DeploymentHealthStore
from litellm.redis.stores.rate_limits import RateLimitStore
from litellm.redis.stores.rate_limits.types import RateLimitScope
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, RouterCooldownStore
from litellm.redis.stores.router_session_pins import DeploymentSessionPin, RouterSessionPinStore
from litellm.redis.stores.router_usage import RouterUsageStore, UsageCounter
from litellm.redis.stores.sensitive_route_pins import SensitiveRoutePinStore
from litellm.redis.stores.spend_counters import SpendCounterStore

_NOW: Final = 1730000000.0

# What each store reads before and after `flush_local()`, and what an empty store reads.
_Reads = tuple[object, object, object]


async def _router_cooldowns() -> _Reads:
    store: Final = RouterCooldownStore(connect(None).router_cooldowns())
    cooldown: Final = CooldownCacheValue(
        exception_received="model", status_code="429", timestamp=_NOW, cooldown_time=120.0
    )
    store.put("dep-a", cooldown, 60.0)
    store.increment_allowed_fails("dep-a", None, 60.0)
    before: Final = (dict(store.snapshot()), store.increment_allowed_fails("dep-b", None, 60.0))
    store.flush_local()
    return (
        before,
        (dict(store.snapshot()), store.increment_allowed_fails("dep-a", None, 60.0)),
        (dict[str, CooldownCacheValue](), 1),
    )


async def _router_usage() -> _Reads:
    store: Final = RouterUsageStore(connect(None).router_usage())
    counter: Final = UsageCounter("global_router", "dep", None, "12-34", "rpm")
    store.set_local(counter, 3)
    before: Final = store.get_local(counter)
    store.flush_local()
    return before, store.get_local(counter), None


async def _rate_limits() -> _Reads:
    store: Final = RateLimitStore(connect(None).rate_limits())
    scope: Final = RateLimitScope("api_key", "k1")
    entries: Final = (scope.window, scope.counter("requests"))
    store.set_local(scope.window, "1730000000", 60)
    store.set_local(scope.counter("requests"), 3, 60)
    before: Final = store.local(entries)
    store.flush_local()
    return before, store.local(entries), (None, None)


async def _router_session_pins() -> _Reads:
    store: Final = RouterSessionPinStore(connect(None).router_session_pins())
    key: Final = DeploymentSessionPin(model_group="group", caller=None, session_id="session")
    store.set_local(key, {"model_id": "dep-a"}, 60)
    before: Final = store.get_local(key)
    store.flush_local()
    return before, store.get_local(key), None


async def _deployment_health() -> _Reads:
    store: Final = DeploymentHealthStore(connect(None).deployment_health())
    store.set_sync({"dep-a": {"is_healthy": False, "timestamp": _NOW, "reason": "timeout"}}, 90.0)
    before: Final = store.get_sync()
    store.flush_local()
    return before, store.get_sync(), None


async def _sensitive_route_pins() -> _Reads:
    store: Final = SensitiveRoutePinStore(connect(None).sensitive_route_pins())
    await store.set("tenant", "session", "model-a", 60)
    before: Final = await store.get("tenant", "session", 60)
    store.flush_local()
    return before, await store.get("tenant", "session", 60), None


async def _active_teams() -> _Reads:
    store: Final = ActiveTeamStore(connect(None).active_teams(clock=lambda: _NOW))
    await store.touch("chat", "team-a", "interactive", 60)
    before: Final = await store.touch("chat", "team-b", "interactive", 60)
    store.flush_local()
    return before, await store.touch("chat", "team-b", "interactive", 60), {"team-b": "interactive"}


async def _auth_cache() -> _Reads:
    store: Final = AuthCacheStore(connect(None).auth_cache(AuthCacheSettings(200, 60)))
    store.set_local(AuthObject.team("t1"), {"team_id": "t1"}, None)
    before: Final = store.get_local(AuthObject.team("t1"))
    store.flush_local()
    return before, store.get_local(AuthObject.team("t1")), None


async def _spend_counters() -> _Reads:
    store: Final = SpendCounterStore(connect(None).spend_counters())
    store.set_local("spend:key:k", 4.0)
    before: Final = store.get_local("spend:key:k")
    store.flush_local()
    return before, store.get_local("spend:key:k"), None


_FLUSHES: Final[Mapping[type, Callable[[], Awaitable[_Reads]]]] = {
    RouterCooldownStore: _router_cooldowns,
    RouterUsageStore: _router_usage,
    RateLimitStore: _rate_limits,
    RouterSessionPinStore: _router_session_pins,
    DeploymentHealthStore: _deployment_health,
    SensitiveRoutePinStore: _sensitive_route_pins,
    ActiveTeamStore: _active_teams,
    AuthCacheStore: _auth_cache,
    SpendCounterStore: _spend_counters,
}

# Publishes and subscribes only, so its `flush_local()` has nothing to drop.
_HOLDS_NOTHING: Final = frozenset({CacheInvalidation})


def _holds_a_native_handle(member: type) -> bool:
    init: Final = member.__dict__.get("__init__")
    return isinstance(init, FunctionType) and "_handle" in init.__code__.co_names


def _stores() -> frozenset[type]:
    modules: Final = pkgutil.walk_packages(litellm.redis.stores.__path__, f"{litellm.redis.stores.__name__}.")
    return frozenset(
        member
        for module in modules
        for _, member in inspect.getmembers(importlib.import_module(module.name), inspect.isclass)
        if member.__module__ == module.name and _holds_a_native_handle(member)
    )


def test_every_store_over_a_native_handle_has_a_flush_case_below() -> None:
    assert _stores() == frozenset(_FLUSHES) | _HOLDS_NOTHING


@pytest.mark.parametrize("store", _FLUSHES, ids=[store.__qualname__ for store in _FLUSHES])
async def test_flush_local_leaves_a_store_reading_as_an_empty_one(store: type) -> None:
    before, after, empty = await _FLUSHES[store]()

    assert before != empty
    assert after == empty
