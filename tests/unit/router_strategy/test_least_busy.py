import asyncio
import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import Final
from unittest.mock import patch

import fakeredis
import fakeredis.aioredis
import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreakerOpenError
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.router_strategy.least_busy import IN_FLIGHT_COUNT_TTL_SECONDS, LeastBusyLoggingHandler
from litellm.types.utils import ModelResponse

GROUP: Final = "least-busy-group"
DEPLOYMENT_A: Final[dict[str, object]] = {"model_info": {"id": "dep-a"}}
DEPLOYMENT_B: Final[dict[str, object]] = {"model_info": {"id": "dep-b"}}
HEALTHY: Final = [DEPLOYMENT_A, DEPLOYMENT_B]
START: Final = 1_700_000_000.0


def _call_kwargs(deployment_id: str, call_id: str = "call-1") -> dict[str, object]:
    return {
        "litellm_call_id": call_id,
        "litellm_params": {"metadata": {"model_group": GROUP}, "model_info": {"id": deployment_id}},
    }


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def redis_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def shared_redis(redis_server: fakeredis.FakeServer) -> Iterator[RedisCache]:
    server: Final = redis_server
    with (
        patch("asyncio.get_running_loop", side_effect=RuntimeError("no running event loop")),
        patch(  # test-quality-ok: RedisCache.__init__ builds its client eagerly, with no injection point
            "litellm._redis.get_redis_client", return_value=fakeredis.FakeRedis(server=server)
        ),
    ):
        cache: Final = RedisCache(host="127.0.0.1", port=6379)
    with patch.object(cache, "init_async_client", return_value=fakeredis.aioredis.FakeRedis(server=server)):
        yield cache


class UnreachableRedis:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def track_in_flight(self, key: str, member: str, started_at: float, max_age: int) -> None:
        raise self.error

    async def async_track_in_flight(self, key: str, member: str, started_at: float, max_age: int) -> None:
        raise self.error

    def untrack_in_flight(self, key: str, member: str) -> None:
        raise self.error

    async def async_untrack_in_flight(self, key: str, member: str) -> None:
        raise self.error

    def batch_count_in_flight(self, key_list: list[str], since: float) -> tuple[int, ...]:
        raise self.error

    async def async_batch_count_in_flight(self, key_list: list[str], since: float) -> tuple[int, ...]:
        raise self.error


def _worker(
    shared: RedisCache | UnreachableRedis | None, clock: Callable[[], float] = time.time
) -> LeastBusyLoggingHandler:
    cache: Final = DualCache(in_memory_cache=InMemoryCache(), redis_cache=shared)  # pyright: ignore[reportArgumentType]  # UnreachableRedis is a duck-typed Redis double
    return LeastBusyLoggingHandler(router_cache=cache, clock=clock)


@pytest.mark.asyncio
async def test_worker_routes_around_a_request_another_worker_started(shared_redis: RedisCache) -> None:
    streaming_worker: Final = _worker(shared_redis)
    picking_worker: Final = _worker(shared_redis)

    picking_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "quick"))
    await picking_worker.async_log_success_event(_call_kwargs("dep-a", "quick"), None, None, None)

    streaming_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "stream"))

    assert await picking_worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await streaming_worker.async_log_success_event(_call_kwargs("dep-a", "stream"), None, None, None)

    assert await picking_worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


def test_sync_pick_reads_the_shared_counts(shared_redis: RedisCache) -> None:
    streaming_worker: Final = _worker(shared_redis)
    picking_worker: Final = _worker(shared_redis)

    picking_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "quick"))
    picking_worker.log_success_event(_call_kwargs("dep-a", "quick"), None, None, None)

    streaming_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "stream"))

    assert picking_worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    streaming_worker.log_failure_event(_call_kwargs("dep-a", "stream"), None, None, None)

    assert picking_worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


@pytest.mark.asyncio
async def test_a_request_outliving_the_ttl_cannot_cancel_a_newer_request_when_it_finishes(
    shared_redis: RedisCache,
) -> None:
    """A shared counter forgot every in-flight request when its key expired, then each late
    decrement cancelled a request that was really running, so a busy deployment read idle and
    kept getting a full share of new traffic (#48994)"""
    clock: Final = Clock()
    worker: Final = _worker(shared_redis, clock)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "old"))
    clock.now = START + IN_FLIGHT_COUNT_TTL_SECONDS + 1
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "new"))
    await worker.async_log_success_event(_call_kwargs("dep-a", "old"), None, None, None)

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B
    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B


@pytest.mark.asyncio
async def test_finishing_one_of_several_calls_sharing_a_call_id_keeps_the_others_counted(
    shared_redis: RedisCache,
) -> None:
    """Callers can pass their own litellm_call_id, so concurrent calls may share one. Tracking the
    id itself merged them into one entry, and the first to finish hid the rest from other workers"""
    worker: Final = _worker(shared_redis)
    other_worker: Final = _worker(shared_redis)
    picking_worker: Final = _worker(shared_redis)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "shared"))
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "shared"))
    other_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "shared"))
    picking_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-b", "b"))
    await worker.async_log_success_event(_call_kwargs("dep-a", "shared"), None, None, None)

    assert await picking_worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    worker.log_success_event(_call_kwargs("dep-a", "shared"), None, None, None)
    await other_worker.async_log_failure_event(_call_kwargs("dep-a", "shared"), None, None, None)

    assert picking_worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


@pytest.mark.asyncio
async def test_a_call_sharing_its_call_id_with_an_earlier_one_keeps_its_own_start_time(
    shared_redis: RedisCache,
) -> None:
    """Callbacks cannot tell which same-id call finished. Keeping the earlier call's start time for
    the one still running made it age out before it was an hour old"""
    clock: Final = Clock()
    worker: Final = _worker(shared_redis, clock)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "shared"))
    clock.now = START + 600
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "shared"))
    await worker.async_log_success_event(_call_kwargs("dep-a", "shared"), None, None, None)
    clock.now = START + IN_FLIGHT_COUNT_TTL_SECONDS + 1

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "shared"))
    worker.log_success_event(_call_kwargs("dep-a", "shared"), None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await worker.async_log_failure_event(_call_kwargs("dep-a", "shared"), None, None, None)

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


@pytest.mark.asyncio
async def test_a_request_nobody_finished_stops_counting_once_it_is_older_than_the_ttl(
    shared_redis: RedisCache,
) -> None:
    """A worker that dies mid-request never reports the end. Its request has to age out on its
    own, without taking the deployment's other in-flight requests with it"""
    clock: Final = Clock()
    worker: Final = _worker(shared_redis, clock)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "orphan"))
    clock.now = START + IN_FLIGHT_COUNT_TTL_SECONDS - 1

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-b", "running-b"))
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a", "running-a"))
    clock.now = START + IN_FLIGHT_COUNT_TTL_SECONDS + 1

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A

    await worker.async_log_success_event(_call_kwargs("dep-b", "running-b"), None, None, None)

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B


@pytest.mark.asyncio
async def test_counts_stay_in_memory_without_redis() -> None:
    worker: Final = _worker(None)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B
    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A
    assert worker.router_cache.get_cache(f"{GROUP}_request_count:dep-a") == 0


@pytest.mark.asyncio
async def test_a_redis_outage_falls_back_to_this_workers_own_counts() -> None:
    worker: Final = _worker(UnreachableRedis(ConnectionError("redis is down")))

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B
    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


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


def test_calls_without_a_deployment_are_ignored(shared_redis: RedisCache, redis_server: fakeredis.FakeServer) -> None:
    worker: Final = _worker(shared_redis)

    worker.log_pre_api_call(model="m", messages=[], kwargs={"litellm_params": {"metadata": None}})
    worker.log_pre_api_call(model="m", messages=[], kwargs={})

    assert fakeredis.FakeRedis(server=redis_server).dbsize() == 0


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_falls_back_without_a_warning_per_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker: Final = _worker(UnreachableRedis(RedisCircuitBreakerOpenError("Redis circuit breaker is open")))

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


async def _open_chat_stream(router: Router) -> CustomStreamWrapper:
    return await router.acompletion(
        model=ROUTER_GROUP, messages=[{"role": "user", "content": "write a poem"}], stream=True
    )


@pytest.mark.asyncio
async def test_router_spreads_open_chat_streams_across_least_busy_deployments(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_chat_stream)
    router: Final = _least_busy_router()

    streams: Final = [await _open_chat_stream(router) for _ in ROUTER_DEPLOYMENT_IDS]

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
    response: Final = await router.acompletion(model=ROUTER_GROUP, messages=[{"role": "user", "content": "hi"}])
    for _ in range(10):
        await asyncio.sleep(0)
        await GLOBAL_LOGGING_WORKER.flush()

    assert deployment["model_info"]["id"] == "1"
    assert isinstance(response, ModelResponse)
    assert response._hidden_params["model_id"] == "1"
    assert route.calls.last.request.headers["authorization"] == "Bearer key-1"
    assert _router_counts(router) == seeded


@pytest.mark.asyncio
async def test_router_spreads_open_streams_across_workers_through_the_shared_counts(
    shared_redis: RedisCache, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    keys_used: Final[list[str]] = []

    def record_key_and_stream(request: httpx.Request) -> httpx.Response:
        keys_used.append(request.headers["authorization"])
        return _chat_stream(request)

    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=record_key_and_stream)
    routers: Final = tuple(_least_busy_router() for _ in ROUTER_DEPLOYMENT_IDS)
    for router in routers:
        router.cache.redis_cache = shared_redis

    for router in routers:
        await _open_chat_stream(router)

    assert sorted(keys_used) == [f"Bearer key-{deployment_id}" for deployment_id in ROUTER_DEPLOYMENT_IDS]
