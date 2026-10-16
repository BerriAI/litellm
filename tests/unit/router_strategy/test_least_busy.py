import asyncio
import json
import logging
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm import Router

from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router_strategy.least_busy import IN_FLIGHT_COUNT_TTL_SECONDS, LeastBusyLoggingHandler
from litellm.types.utils import ModelResponse

GROUP: Final = "least-busy-group"
DEPLOYMENT_A: Final[dict[str, object]] = {"model_info": {"id": "dep-a"}}
DEPLOYMENT_B: Final[dict[str, object]] = {"model_info": {"id": "dep-b"}}
HEALTHY: Final = [DEPLOYMENT_A, DEPLOYMENT_B]


def _call_kwargs(deployment_id: str) -> dict[str, object]:
    return {"litellm_params": {"metadata": {"model_group": GROUP}, "model_info": {"id": deployment_id}}}


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
