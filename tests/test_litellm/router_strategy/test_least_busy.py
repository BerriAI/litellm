import asyncio
import collections
import inspect
from datetime import UTC, datetime

import pytest

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler

HEALTHY = [{"model_info": {"id": "A"}}, {"model_info": {"id": "B"}}]
MODEL_GROUP = "grp"
RELEASE_METHODS = ("log_success_event", "log_failure_event", "async_log_success_event", "async_log_failure_event")


def _pick_counts(request_count: dict, healthy: list = HEALTHY, n: int = 1000) -> collections.Counter:
    handler = LeastBusyLoggingHandler(router_cache=DualCache())
    return collections.Counter(
        handler._get_available_deployments(healthy_deployments=healthy, all_deployments=dict(request_count))[
            "model_info"
        ]["id"]
        for _ in range(n)
    )


@pytest.mark.parametrize("request_count", [{}, {"A": 0, "B": 0}, {"A": 7, "B": 7}])
def test_ties_are_spread_across_the_tied_deployments(request_count: dict) -> None:
    picks = _pick_counts(request_count)

    assert set(picks) == {"A", "B"}
    assert min(picks.values()) > 300


def test_a_three_way_tie_reaches_every_deployment() -> None:
    healthy = [{"model_info": {"id": name}} for name in ("A", "B", "C")]

    picks = _pick_counts({}, healthy=healthy, n=1200)

    assert set(picks) == {"A", "B", "C"}
    assert min(picks.values()) > 250


@pytest.mark.parametrize(
    "request_count, expected",
    [
        ({"A": 5, "B": 1}, "B"),
        ({"A": 1, "B": 5}, "A"),
        ({"A": 0, "B": 3}, "A"),
    ],
)
def test_the_least_busy_deployment_wins(request_count: dict, expected: str) -> None:
    assert set(_pick_counts(request_count, n=200)) == {expected}


def test_a_deployment_in_cooldown_cannot_win_the_minimum() -> None:
    picks = _pick_counts({"A": 5, "B": 3, "C": 0}, n=500)

    assert set(picks) == {"B"}


def test_a_deployment_with_no_recorded_traffic_counts_as_idle() -> None:
    picks = _pick_counts({"A": 4}, n=200)

    assert set(picks) == {"B"}


def _count_key(deployment_id: str) -> str:
    return f"{MODEL_GROUP}_request_count:{deployment_id}"


def _kwargs(deployment_id: str = "A", **logging_fields: object) -> dict:
    return {
        "litellm_params": {"metadata": {"model_group": MODEL_GROUP}, "model_info": {"id": deployment_id}},
        **logging_fields,
    }


def _upstream_kwargs(deployment_id: str = "A") -> dict:
    return _kwargs(deployment_id, api_call_start_time=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), cache_hit=None)


def _handler(counts: dict | None = None) -> tuple[LeastBusyLoggingHandler, DualCache]:
    cache = DualCache()
    for deployment_id, count in (counts or {}).items():
        cache.set_cache(key=_count_key(deployment_id), value=count)
    return LeastBusyLoggingHandler(router_cache=cache), cache


def _counts(cache: DualCache, *deployment_ids: str) -> dict:
    return {deployment_id: cache.get_cache(key=_count_key(deployment_id)) or 0 for deployment_id in deployment_ids}


async def _release(handler: LeastBusyLoggingHandler, method: str, kwargs: dict) -> None:
    result = getattr(handler, method)(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    if inspect.isawaitable(result):
        await result


@pytest.mark.parametrize("method", RELEASE_METHODS)
@pytest.mark.asyncio
async def test_a_provider_call_takes_and_releases_one_slot(method: str) -> None:
    handler, cache = _handler()

    handler.log_pre_api_call(model="m", messages=[], kwargs=_kwargs("A"))
    assert _counts(cache, "A") == {"A": 1}

    await _release(handler, method, _upstream_kwargs("A"))
    assert _counts(cache, "A") == {"A": 0}


CACHE_HIT_SHAPES = {
    "provider never called": _kwargs("A", cache_hit=True),
    "an earlier attempt reached a provider": _upstream_kwargs("A") | {"cache_hit": True},
}


@pytest.mark.parametrize("method", RELEASE_METHODS)
@pytest.mark.parametrize("kwargs", CACHE_HIT_SHAPES.values(), ids=CACHE_HIT_SHAPES.keys())
@pytest.mark.asyncio
async def test_a_cache_hit_does_not_release_a_slot(method: str, kwargs: dict) -> None:
    handler, cache = _handler({"A": 1, "B": 1})

    await _release(handler, method, kwargs)

    assert _counts(cache, "A", "B") == {"A": 1, "B": 1}


@pytest.mark.parametrize("method", ("log_failure_event", "async_log_failure_event"))
@pytest.mark.asyncio
async def test_a_failure_before_the_provider_call_does_not_release_a_slot(method: str) -> None:
    handler, cache = _handler({"A": 1})

    await _release(handler, method, _kwargs("A"))

    assert _counts(cache, "A") == {"A": 1}


@pytest.mark.parametrize("method", RELEASE_METHODS)
@pytest.mark.parametrize("seed", (0, -5))
@pytest.mark.asyncio
async def test_a_slot_count_never_stays_below_zero(method: str, seed: int) -> None:
    handler, cache = _handler({"A": seed})

    await _release(handler, method, _upstream_kwargs("A"))

    assert _counts(cache, "A") == {"A": 0}


def _deployment(deployment_id: str) -> dict:
    return {
        "model_name": MODEL_GROUP,
        "litellm_params": {"model": "openai/fake", "api_key": "x", "mock_response": "ok"},
        "model_info": {"id": deployment_id},
    }


@pytest.mark.asyncio
async def test_cache_hits_do_not_starve_a_deployment() -> None:
    router = Router(
        model_list=[_deployment("A"), _deployment("B")], routing_strategy="least-busy", cache_responses=True
    )
    messages = [{"role": "user", "content": "same prompt every time"}]

    miss = await router.acompletion(model=MODEL_GROUP, messages=messages)
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    assert miss._hidden_params.get("cache_hit") is not True
    assert _counts(router.cache, "A", "B") == {"A": 0, "B": 0}

    for deployment_id in ("A", "B"):
        await router.cache.async_set_cache(key=_count_key(deployment_id), value=1)

    hits = [await router.acompletion(model=MODEL_GROUP, messages=messages) for _ in range(5)]
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    assert all(hit._hidden_params.get("cache_hit") is True for hit in hits)
    assert _counts(router.cache, "A", "B") == {"A": 1, "B": 1}

    picks = [
        (await router.async_get_available_deployment(model=MODEL_GROUP, request_kwargs={}, messages=messages))[
            "model_info"
        ]["id"]
        for _ in range(100)
    ]
    assert set(picks) == {"A", "B"}


class SharedStore:
    def __init__(self) -> None:
        self.values: dict = {}

    def increment_cache(self, key: str, value: int, **kwargs: object) -> int:
        self.values[key] = self.values.get(key, 0) + value
        return self.values[key]

    async def async_increment(self, key: str, value: int, **kwargs: object) -> int:
        return self.increment_cache(key, value)

    def set_cache(self, key: str, value: int, **kwargs: object) -> None:
        self.values[key] = value

    async def async_set_cache(self, key: str, value: int, **kwargs: object) -> None:
        self.values[key] = value

    def batch_get_cache(self, key_list: list[str], **kwargs: object) -> dict:
        return {key: self.values.get(key) for key in key_list}

    async def async_batch_get_cache(self, key_list: list[str], **kwargs: object) -> dict:
        return self.batch_get_cache(key_list)


def _worker(store: SharedStore) -> LeastBusyLoggingHandler:
    return LeastBusyLoggingHandler(router_cache=DualCache(in_memory_cache=InMemoryCache(), redis_cache=store))


def _pick(worker: LeastBusyLoggingHandler) -> str:
    return worker.get_available_deployments(model_group=MODEL_GROUP, healthy_deployments=HEALTHY)["model_info"]["id"]


async def _async_pick(worker: LeastBusyLoggingHandler) -> str:
    chosen = await worker.async_get_available_deployments(model_group=MODEL_GROUP, healthy_deployments=HEALTHY)
    return chosen["model_info"]["id"]


@pytest.mark.asyncio
async def test_in_flight_counts_are_shared_across_workers() -> None:
    store = SharedStore()
    worker_one, worker_two = _worker(store), _worker(store)

    for _ in range(3):
        worker_one.log_pre_api_call(model="m", messages=[], kwargs=_kwargs("A"))
    assert {await _async_pick(worker_two) for _ in range(50)} == {"B"}

    worker_two.log_pre_api_call(model="m", messages=[], kwargs=_kwargs("B"))
    for _ in range(3):
        await worker_one.async_log_success_event(
            _upstream_kwargs("A"), response_obj=None, start_time=None, end_time=None
        )

    assert store.values == {_count_key("A"): 0, _count_key("B"): 1}
    assert {_pick(worker_one) for _ in range(50)} == {"A"}
    assert {await _async_pick(worker_two) for _ in range(50)} == {"A"}
