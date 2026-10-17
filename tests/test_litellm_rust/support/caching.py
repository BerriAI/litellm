import asyncio
import threading
from collections.abc import Callable, Coroutine, Generator
from dataclasses import dataclass
from typing import Final, TypeAlias
from urllib.parse import urlparse

import fakeredis
import pytest
import redis
from pydantic import TypeAdapter

import litellm
from litellm.caching.base_cache import BaseCache
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await the existing background cache writer before the next request
    in_memory_cache_obj,
)
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.hooks.model_max_budget_limiter import (
    PROXY_VirtualKeyModelMaxBudgetLimiter,
    model_budget_spend_cache_key,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.utils import InternalUsageCache
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer

Call: TypeAlias = Callable[..., Coroutine[object, object, object]]


class CallerCache(BaseCache):
    def __init__(self, lookup_entered: asyncio.Event | None = None, release: asyncio.Event | None = None) -> None:
        super().__init__()
        self.storage: Final = InMemoryCache()
        self.lookup_tasks: Final[list[asyncio.Task[object] | None]] = []  # mutable-ok: test double records each lookup
        self.lookup_entered: Final = lookup_entered
        self.release: Final = release

    async def async_set_cache_pipeline(self, cache_list: list[tuple[str, object]], ttl: float | None = None) -> None:
        await self.storage.async_set_cache_pipeline(cache_list, ttl=ttl)

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        self.lookup_tasks.append(asyncio.current_task())
        if self.lookup_entered is not None and self.release is not None:
            self.lookup_entered.set()
            await self.release.wait()
        return self.storage.get_cache(key, **kwargs)

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        await asyncio.sleep(0)
        self.storage.set_cache(key, value, **kwargs)


class HeldWriteCache(InMemoryCache):
    def __init__(self) -> None:
        super().__init__()
        self.release: Final = asyncio.Event()

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        await self.release.wait()
        await super().async_set_cache(key, value, **kwargs)


class FailingLookupCache(Cache):
    def should_use_cache(self, **kwargs: object) -> bool:
        raise RuntimeError("cache lookup failed")


def use_cache(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Cache:
    monkeypatch.setattr(litellm, "cache", cache)
    return cache


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> Cache:
    return use_cache(monkeypatch, Cache())


@pytest.fixture
def call(native_call: Call) -> Call:
    async def send(**kwargs: object) -> object:
        response: Final = await native_call(**kwargs)
        await settle_cache_writes()
        return response

    return send


@dataclass(frozen=True, slots=True)
class FakeRedis:
    url: str
    client: redis.Redis

    def cache(self) -> Cache:
        parsed: Final = urlparse(self.url)
        in_memory_cache_obj.flush_cache()
        return Cache(type=LiteLLMCacheType.REDIS, host=parsed.hostname, port=str(parsed.port))


@pytest.fixture
def fake_redis() -> Generator[FakeRedis]:
    server: Final = fakeredis.TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    worker: Final = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url: Final = f"redis://127.0.0.1:{server.server_address[1]}"
    try:
        yield FakeRedis(url=url, client=redis.Redis.from_url(url))
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


async def settle_cache_writes() -> None:
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))


async def assert_cache_hit_keeps_budget_spend_and_counts_usage(
    send: Call, server: RecordingServer, tokens: int
) -> None:
    counters: Final = litellm.DualCache()
    budget: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(counters)
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        InternalUsageCache(counters), model_group_resolver=lambda model: model
    )
    recorder: Final = RecordingLogger()
    key_hash: Final = "a" * 64
    options: Final = {
        "callbacks": [budget, limiter, recorder],
        "metadata": {
            "user_api_key": key_hash,
            "model_group": "cached-model",
            "user_api_key_model_max_budget": {"cached-model": {"max_budget": 1, "budget_duration": "1h"}},
        },
    }
    spend_key: Final = model_budget_spend_cache_key(Litellm_EntityType.KEY, key_hash, "cached-model", "1h")
    token_key: Final = limiter.create_rate_limit_keys("api_key", key_hash, "tokens")
    objects: Final = TypeAdapter(dict[str, object])

    await send(**options)
    await drain_logging()
    miss: Final = objects.validate_python((await recorder.wait_for_async("async_log_success_event"))[0].kwargs)
    cost: Final = TypeAdapter(float).validate_python(miss["response_cost"])
    assert cost > 0
    assert counters.get_cache(spend_key) == pytest.approx(cost)
    assert counters.get_cache(token_key) == tokens

    await send(**options)
    await drain_logging()
    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    hit: Final = objects.validate_python(successes[-1].kwargs)
    miss_payload: Final = objects.validate_python(miss["standard_logging_object"])
    hit_payload: Final = objects.validate_python(hit["standard_logging_object"])
    assert len(server.requests) == 1
    assert hit["cache_hit"] is True
    assert hit["response_cost"] == hit_payload["response_cost"] == 0
    assert hit_payload["custom_llm_provider"] == miss_payload["custom_llm_provider"]
    assert hit_payload["total_tokens"] == miss_payload["total_tokens"] == tokens
    assert counters.get_cache(spend_key) == pytest.approx(cost)
    assert counters.get_cache(token_key) == 2 * tokens
