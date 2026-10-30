import asyncio
import contextlib
import datetime
import json
import logging
import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import httpx
import openai
import pytest
import respx

import litellm
from litellm import Router

from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import DEFAULT_MAX_SIZE_IN_MEMORY, InMemoryCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router_strategy.least_busy import (
    IN_FLIGHT_COUNT_TTL_SECONDS,
    SELECTION_RESERVATION_TTL_SECONDS,
    LeastBusyLoggingHandler,
    picks_reserve,
)
from litellm.types.utils import ModelResponse

GROUP: Final = "least-busy-group"
DEPLOYMENT_A: Final[dict[str, object]] = {"model_info": {"id": "dep-a"}}
DEPLOYMENT_B: Final[dict[str, object]] = {"model_info": {"id": "dep-b"}}
HEALTHY: Final = [DEPLOYMENT_A, DEPLOYMENT_B]


def _call_kwargs(deployment_id: str, call_id: str = "call-1") -> dict[str, object]:
    return {
        "litellm_params": {"metadata": {"model_group": GROUP}, "model_info": {"id": deployment_id}},
        "litellm_call_id": call_id,
    }


class SharedRedisCounters:
    """Mirrors what Redis gives the handler: increments clamped at zero, a TTL set once when
    the key is created, and ordered reads that raise rather than invent a value."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    def count(self, key: str) -> int | None:
        return self.counts.get(key)

    def expire(self, key: str) -> None:
        self.counts.pop(key, None)
        self.ttls.pop(key, None)

    def increment_with_floor(self, key: str, value: int, ttl: int) -> int:
        incremented: Final = max(0, self.counts.get(key, 0) + value)
        self.counts[key] = incremented
        self.ttls.setdefault(key, ttl)
        return incremented

    async def async_increment_with_floor(self, key: str, value: int, ttl: int) -> int:
        return self.increment_with_floor(key, value, ttl)

    def batch_get_counts(self, key_list: list[str]) -> tuple[int | None, ...]:
        return tuple(self.counts.get(key) for key in key_list)

    async def async_batch_get_counts(self, key_list: list[str]) -> tuple[int | None, ...]:
        return self.batch_get_counts(key_list)


class SlowRedisRead(SharedRedisCounters):
    """Answers a count read with the counts from when the read started, after letting other work run."""

    def __init__(self) -> None:
        super().__init__()
        self.while_reading: Callable[[], None] = lambda: None

    async def async_batch_get_counts(self, key_list: list[str]) -> tuple[int | None, ...]:
        counts_when_asked: Final = self.batch_get_counts(key_list)
        await asyncio.sleep(0)
        self.while_reading()
        return counts_when_asked


def _worker(shared: SharedRedisCounters | None) -> LeastBusyLoggingHandler:
    cache: Final = DualCache(in_memory_cache=InMemoryCache(), redis_cache=shared)  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    return LeastBusyLoggingHandler(router_cache=cache)


@pytest.mark.asyncio
async def test_worker_routes_around_a_request_another_worker_started() -> None:
    shared: Final = SharedRedisCounters()
    streaming_worker: Final = _worker(shared)
    picking_worker: Final = _worker(shared)

    picking_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    await picking_worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    streaming_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert await picking_worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await streaming_worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert await picking_worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


@pytest.mark.asyncio
async def test_a_request_that_starts_while_redis_is_read_still_counts_on_its_deployment() -> None:
    redis: Final = SlowRedisRead()
    worker: Final = _worker(redis)
    assert await worker.async_get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    redis.while_reading = lambda: worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert await worker.async_get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B


def test_sync_pick_reads_the_shared_counts() -> None:
    shared: Final = SharedRedisCounters()
    streaming_worker: Final = _worker(shared)
    picking_worker: Final = _worker(shared)

    picking_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    picking_worker.log_success_event(_call_kwargs("dep-a"), None, None, None)

    streaming_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert picking_worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    streaming_worker.log_failure_event(_call_kwargs("dep-a"), None, None, None)

    assert picking_worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


def _routed_through_litellm_metadata(deployment_id: str) -> dict[str, object]:
    return {
        "litellm_params": {
            "metadata": {"tags": ["caller-tag"]},
            "litellm_metadata": {"model_group": GROUP},
            "model_info": {"id": deployment_id},
        },
        "litellm_call_id": "call-1",
    }


def test_a_request_routed_through_litellm_metadata_counts_while_in_flight() -> None:
    shared: Final = SharedRedisCounters()
    calling_worker: Final = _worker(shared)
    picking_worker: Final = _worker(shared)

    assert calling_worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    calling_worker.log_pre_api_call(model="m", messages=[], kwargs=_routed_through_litellm_metadata("dep-a"))

    assert picking_worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    calling_worker.log_success_event(_routed_through_litellm_metadata("dep-a"), None, None, None)

    assert calling_worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


def test_the_handler_never_pushes_a_counters_ttl_forward() -> None:
    """A worker that dies mid-request leaves a +1 nobody will ever decrement. Redis expires that
    stuck count an hour after the key was created, which only works while nothing writes the TTL
    again: a handler that refreshed it on every touch would keep the count alive for as long as
    the group takes traffic, and the deployment would read busier than it is forever."""
    shared: Final = SharedRedisCounters()
    worker: Final = _worker(shared)
    key: Final = f"{GROUP}_request_count:dep-a"

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert shared.ttls == {key: IN_FLIGHT_COUNT_TTL_SECONDS}

    shared.ttls[key] = 5
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    worker.log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert shared.count(key) == 1
    assert shared.ttls == {key: 5}


@pytest.mark.asyncio
async def test_counts_stay_in_memory_without_redis() -> None:
    worker: Final = _worker(None)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B
    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A
    assert worker.router_cache.get_cache(f"{GROUP}_request_count:dep-a") == 0


class UnavailableRedis(SharedRedisCounters):
    def increment_with_floor(self, key: str, value: int, ttl: int) -> int:
        raise ConnectionError("redis is down")

    def batch_get_counts(self, key_list: list[str]) -> tuple[int | None, ...]:
        raise ConnectionError("redis is down")


@pytest.mark.asyncio
async def test_a_redis_outage_falls_back_to_this_workers_own_counts() -> None:
    worker: Final = _worker(UnavailableRedis())

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B
    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


def test_a_shared_counter_that_expired_mid_request_cannot_go_negative() -> None:
    shared: Final = SharedRedisCounters()
    worker: Final = _worker(shared)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    shared.expire(f"{GROUP}_request_count:dep-a")
    worker.log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert shared.count(f"{GROUP}_request_count:dep-a") == 0

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert shared.count(f"{GROUP}_request_count:dep-a") == 1
    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B


@pytest.mark.asyncio
async def test_a_local_counter_that_expired_mid_request_cannot_go_negative() -> None:
    worker: Final = _worker(None)
    in_memory: Final = worker.router_cache.in_memory_cache

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    in_memory.delete_cache(f"{GROUP}_request_count:dep-a")
    await worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert worker.router_cache.get_cache(f"{GROUP}_request_count:dep-a") == 0

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B


def test_calls_without_a_deployment_are_ignored() -> None:
    shared: Final = SharedRedisCounters()
    worker: Final = _worker(shared)

    worker.log_pre_api_call(model="m", messages=[], kwargs={"litellm_params": {"metadata": None}})
    worker.log_pre_api_call(model="m", messages=[], kwargs={})

    assert shared.counts == {}


class OpenBreakerRedis(SharedRedisCounters):
    def increment_with_floor(self, key: str, value: int, ttl: int) -> int:
        raise RedisCircuitBreakerOpenError("Redis circuit breaker is open")

    def batch_get_counts(self, key_list: list[str]) -> tuple[int | None, ...]:
        raise RedisCircuitBreakerOpenError("Redis circuit breaker is open")


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_falls_back_without_a_warning_per_request(caplog: pytest.LogCaptureFixture) -> None:
    worker: Final = _worker(OpenBreakerRedis())

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Router"):
        worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
        picked: Final = worker.get_available_deployments(GROUP, HEALTHY)

    assert picked is DEPLOYMENT_B
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []
    assert sum("circuit breaker is open" in record.getMessage() for record in caplog.records) == 2


def test_model_added():
    test_cache = DualCache()
    least_busy_logger = LeastBusyLoggingHandler(router_cache=test_cache)
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": "1234"},
        }
    }
    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    request_count_api_key = "gpt-3.5-turbo_request_count:1234"
    assert test_cache.get_cache(key=request_count_api_key) == 1


def test_get_available_deployments():
    test_cache = DualCache()
    least_busy_logger = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "gpt-3.5-turbo"
    deployment = "azure/gpt-4.1-mini"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": deployment,
            },
            "model_info": {"id": "1234"},
        }
    }
    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    request_count_api_key = f"{model_group}_request_count:1234"
    assert test_cache.get_cache(key=request_count_api_key) == 1


ROUTER_GROUP: Final = "least-busy-router-group"
ROUTER_DEPLOYMENT_IDS: Final = ("1", "2", "3")


def _least_busy_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": ROUTER_GROUP,
                "litellm_params": {"model": "openai/gpt-4.1-mini", "api_key": f"key-{deployment_id}"},
                "model_info": {"id": deployment_id},
            }
            for deployment_id in ROUTER_DEPLOYMENT_IDS
        ],
        routing_strategy="least-busy",
        num_retries=0,
    )


def _router_counts(router: Router) -> dict[str, object]:
    return {
        deployment_id: router.cache.get_cache(key=f"{ROUTER_GROUP}_request_count:{deployment_id}")
        for deployment_id in ROUTER_DEPLOYMENT_IDS
    }


def _sse(chunks: tuple[dict[str, object], ...]) -> httpx.Response:
    body: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def _chat_stream(_: httpx.Request) -> httpx.Response:
    return _sse(
        (
            {
                "id": "chatcmpl-least-busy",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4.1-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "poem"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-least-busy",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4.1-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        )
    )


@pytest.mark.asyncio
async def test_router_spreads_open_chat_streams_across_least_busy_deployments(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_chat_stream)
    router: Final = _least_busy_router()

    streams: Final = [
        await router.acompletion(
            model=ROUTER_GROUP, messages=[{"role": "user", "content": "write a poem"}], stream=True
        )
        for _ in ROUTER_DEPLOYMENT_IDS
    ]

    assert _router_counts(router) == {"1": 1, "2": 1, "3": 1}
    assert sorted(stream._hidden_params["model_id"] for stream in streams) == list(ROUTER_DEPLOYMENT_IDS)
    assert sorted(call.request.headers["authorization"] for call in route.calls) == [
        "Bearer key-1",
        "Bearer key-2",
        "Bearer key-3",
    ]


@pytest.mark.asyncio
async def test_router_spreads_open_text_completion_streams_across_least_busy_deployments(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_chat_stream)
    router: Final = _least_busy_router()

    for _ in ROUTER_DEPLOYMENT_IDS:
        await router.atext_completion(model=ROUTER_GROUP, prompt="write a poem", stream=True)

    assert _router_counts(router) == {"1": 1, "2": 1, "3": 1}
    assert sorted(call.request.headers["authorization"] for call in route.calls) == [
        "Bearer key-1",
        "Bearer key-2",
        "Bearer key-3",
    ]


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.asyncio
async def test_router_picks_least_busy_deployment_and_completion_restores_counts(
    use_async: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-least-busy",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4.1-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    router: Final = _least_busy_router()
    seeded: Final = {"1": 10, "2": 54, "3": 100}
    for deployment_id, count in seeded.items():
        router.cache.set_cache(key=f"{ROUTER_GROUP}_request_count:{deployment_id}", value=count)

    deployment: Final = (
        await router.async_get_available_deployment(model=ROUTER_GROUP, messages=None, request_kwargs={})
        if use_async
        else router.get_available_deployment(model=ROUTER_GROUP, messages=None)
    )
    response: Final = await router.acompletion(
        model=ROUTER_GROUP, messages=[{"role": "user", "content": "hi"}]
    )
    for _ in range(10):
        await asyncio.sleep(0)
        await GLOBAL_LOGGING_WORKER.flush()

    assert deployment["model_info"]["id"] == "1"
    assert isinstance(response, ModelResponse)
    assert response._hidden_params["model_id"] == "1"
    assert route.calls.last.request.headers["authorization"] == "Bearer key-1"
    assert _router_counts(router) == seeded


@pytest.mark.asyncio
async def test_router_spreads_requests_that_start_together(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    burst: Final = 2 * len(ROUTER_DEPLOYMENT_IDS)
    all_started: Final = asyncio.Barrier(burst)

    async def completion_after_all_started(_: httpx.Request) -> httpx.Response:
        await all_started.wait()
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-least-busy",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4.1-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=completion_after_all_started
    )
    router: Final = _least_busy_router()

    await asyncio.gather(
        *(router.acompletion(model=ROUTER_GROUP, messages=[{"role": "user", "content": "hi"}]) for _ in range(burst))
    )

    one_per_deployment: Final = tuple(f"Bearer key-{deployment_id}" for deployment_id in ROUTER_DEPLOYMENT_IDS)
    assert sorted(call.request.headers["authorization"] for call in route.calls) == sorted(one_per_deployment * 2)


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_reservation_that_never_reaches_the_api_call_expires() -> None:
    clock: Final = _ManualClock()
    worker: Final = LeastBusyLoggingHandler(router_cache=DualCache(), clock=clock)

    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    clock.now = 1.0
    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B
    clock.now = 2.0
    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    clock.now = SELECTION_RESERVATION_TTL_SECONDS + 0.5
    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A


def test_a_fresh_reservation_outlives_the_ttl_of_the_first_one_on_its_deployment() -> None:
    clock: Final = _ManualClock()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock))
    worker: Final = LeastBusyLoggingHandler(router_cache=cache, clock=clock)

    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    worker.log_success_event(kwargs=_call_kwargs("dep-a"), response_obj=None, start_time=None, end_time=None)
    clock.now = SELECTION_RESERVATION_TTL_SECONDS - 0.1
    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    clock.now = SELECTION_RESERVATION_TTL_SECONDS + 0.1

    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B


def test_a_started_request_moves_its_reservation_into_the_in_flight_count() -> None:
    worker: Final = _worker(None)

    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B
    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A


def test_a_request_started_through_another_selector_of_the_router_releases_the_reservation() -> None:
    cache: Final = DualCache()
    reservations: Final = InMemoryCache()
    picker: Final = LeastBusyLoggingHandler(router_cache=cache, reservation_store=reservations)
    registered: Final = LeastBusyLoggingHandler(router_cache=cache, reservation_store=reservations)

    assert picker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    registered.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert picker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B
    assert picker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A


class _StoreWhereThreadsReadBeforeWriting:
    def __init__(self) -> None:
        self._store: Final = InMemoryCache()
        self._first_reads: Final = threading.Barrier(2, timeout=0.5)
        self._thread_state: Final = threading.local()
        self._meeting: Final = threading.Event()

    @contextlib.contextmanager
    def first_reads_meet(self) -> Generator[None, None, None]:
        self._meeting.set()
        yield
        self._meeting.clear()

    def get_cache(self, key: str) -> object:
        value: Final = self._store.get_cache(key)
        if self._meeting.is_set() and not getattr(self._thread_state, "has_read", False):
            self._thread_state.has_read = True
            with contextlib.suppress(threading.BrokenBarrierError):
                self._first_reads.wait()
        return value

    def set_cache(self, key: str, value: tuple[float, ...], *, ttl: float) -> None:
        self._store.set_cache(key, value, ttl=ttl)

    def delete_cache(self, key: str) -> None:
        self._store.delete_cache(key)


def test_two_threads_picking_at_once_reserve_different_deployments() -> None:
    store: Final = _StoreWhereThreadsReadBeforeWriting()
    worker: Final = LeastBusyLoggingHandler(router_cache=DualCache(), reservation_store=store)

    with store.first_reads_meet(), ThreadPoolExecutor(max_workers=2) as pool:
        picks: Final = tuple(pool.map(lambda _: worker.get_available_deployments(GROUP, HEALTHY, reserve=True), range(2)))

    assert sorted(pick["model_info"]["id"] for pick in picks if pick is not None) == ["dep-a", "dep-b"]


def test_a_reservation_and_a_release_on_one_deployment_at_once_both_count() -> None:
    store: Final = _StoreWhereThreadsReadBeforeWriting()
    worker: Final = LeastBusyLoggingHandler(router_cache=DualCache(), reservation_store=store)
    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-b"))
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-b"))

    with store.first_reads_meet(), ThreadPoolExecutor(max_workers=2) as pool:
        reserved: Final = pool.submit(worker.get_available_deployments, GROUP, HEALTHY, True)
        started: Final = pool.submit(worker.log_pre_api_call, "m", [], _call_kwargs("dep-a"))
        assert reserved.result() is DEPLOYMENT_A
        started.result()

    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B


def _router_pick(router: Router) -> str:
    deployment: Final = router.get_available_deployment(ROUTER_GROUP, request_kwargs={"metadata": {}})
    return deployment["model_info"]["id"]


def test_a_pick_stays_reserved_while_other_router_keys_fill_its_cache() -> None:
    router: Final = _least_busy_router()

    with picks_reserve(True):
        first: Final = _router_pick(router)
        for index in range(DEFAULT_MAX_SIZE_IN_MEMORY):
            router.health().store.set_cache(f"deployment-usage-{index}", 1, local_only=True, ttl=60)

        assert _router_pick(router) != first


@pytest.mark.asyncio
async def test_looking_up_a_deployment_does_not_hold_it_for_later_picks() -> None:
    router: Final = _least_busy_router()

    sync_picks: Final = [_router_pick(router) for _ in range(3)]
    async_picks: Final = [
        (await router.async_get_available_deployment(ROUTER_GROUP, request_kwargs={"metadata": {}}))["model_info"]["id"]
        for _ in range(3)
    ]

    assert sync_picks + async_picks == [ROUTER_DEPLOYMENT_IDS[0]] * 6


@pytest.mark.asyncio
async def test_a_cache_hit_releases_the_pick_it_reserved() -> None:
    worker: Final = _worker(None)
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert await worker.async_get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B
    await worker.async_log_success_event({**_call_kwargs("dep-b"), "cache_hit": True}, None, None, None)

    assert await worker.async_get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B


def test_a_cache_hit_leaves_requests_in_flight_on_its_deployment_counted() -> None:
    worker: Final = _worker(None)
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    worker.log_success_event({**_call_kwargs("dep-a", "cached-call"), "cache_hit": True}, None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B


@pytest.mark.parametrize("outcome", [LeastBusyLoggingHandler.log_success_event, LeastBusyLoggingHandler.log_failure_event])
def test_a_call_that_ends_before_reaching_the_provider_frees_its_pick(
    outcome: Callable[[LeastBusyLoggingHandler, dict[str, object], None, None, None], None],
) -> None:
    worker: Final = _worker(None)
    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A

    outcome(worker, _call_kwargs("dep-a"), None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


def test_a_call_the_proxy_rejected_after_logging_its_start_leaves_no_count_behind() -> None:
    worker: Final = _worker(None)
    rejected: Final = {**_call_kwargs("dep-a"), LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL: True}

    worker.log_pre_api_call(model="m", messages=[], kwargs=rejected)
    worker.log_failure_event(rejected, None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


def test_a_call_logged_under_its_callers_group_name_frees_the_pick_made_for_another_group() -> None:
    worker: Final = _worker(None)
    assert worker.get_available_deployments("tier-group", HEALTHY, reserve=True) is DEPLOYMENT_A

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    worker.log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert worker.get_available_deployments("tier-group", HEALTHY) is DEPLOYMENT_A


def test_a_fallback_attempt_that_fails_before_reaching_its_provider_frees_its_pick() -> None:
    worker: Final = _worker(None)
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "busy-call"))
    first_attempt_started_at: Final = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    first_attempt: Final = {**_call_kwargs("dep-a", "fallback-call"), "api_call_start_time": first_attempt_started_at}
    worker.log_pre_api_call(model="m", messages=[], kwargs=first_attempt)
    worker.log_failure_event(first_attempt, None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B
    second_attempt: Final = {**_call_kwargs("dep-b", "fallback-call"), "api_call_start_time": first_attempt_started_at}
    worker.log_failure_event(second_attempt, None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B


def test_a_retry_that_starts_before_the_failed_attempt_is_logged_counts_each_attempt_once() -> None:
    worker: Final = _worker(None)
    attempt: Final = _call_kwargs("dep-a", "retried-call")

    worker.log_pre_api_call(model="m", messages=[], kwargs=attempt)
    worker.log_pre_api_call(model="m", messages=[], kwargs=attempt)
    worker.log_failure_event(attempt, None, None, None)
    worker.log_success_event(attempt, None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_A
    assert worker.get_available_deployments(GROUP, HEALTHY, reserve=True) is DEPLOYMENT_B


async def _drain_logging() -> None:
    for _ in range(10):
        await asyncio.sleep(0)
        await GLOBAL_LOGGING_WORKER.flush()


def _batch_deployment(route: respx.Route) -> str:
    return route.calls.last.request.headers["authorization"].removeprefix("Bearer key-")


_BATCH: Final = {
    "id": "batch_least_busy",
    "object": "batch",
    "endpoint": "/v1/chat/completions",
    "input_file_id": "file-least-busy",
    "completion_window": "24h",
    "status": "validating",
    "created_at": 1,
}
_REJECTED_BATCH: Final = {"error": {"message": "Invalid input file", "type": "invalid_request_error"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_response", [httpx.Response(200, json=_BATCH), httpx.Response(400, json=_REJECTED_BATCH)])
async def test_a_batch_create_does_not_keep_its_deployment_busy(
    provider_response: httpx.Response, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/batches").mock(return_value=provider_response)
    router: Final = _least_busy_router()

    with contextlib.suppress(openai.BadRequestError):
        await router.acreate_batch(
            model=ROUTER_GROUP, completion_window="24h", endpoint="/v1/chat/completions", input_file_id="file-least-busy"
        )
    await _drain_logging()

    assert _router_pick(router) == _batch_deployment(route)


@pytest.mark.asyncio
async def test_a_batch_cancel_does_not_keep_its_deployment_busy(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/batches/batch_least_busy/cancel").mock(
        return_value=httpx.Response(200, json={**_BATCH, "status": "cancelling"})
    )
    router: Final = _least_busy_router()

    await router.acancel_batch(model=ROUTER_GROUP, batch_id="batch_least_busy")

    assert _router_pick(router) == _batch_deployment(route)


@pytest.mark.asyncio
async def test_picking_a_deployment_without_calling_it_does_not_keep_it_busy() -> None:
    router: Final = _least_busy_router()

    picked: Final = await router.select(ROUTER_GROUP, {"metadata": {}})

    assert _router_pick(router) == picked["model_info"]["id"]
