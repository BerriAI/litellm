from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone, tzinfo
from types import SimpleNamespace
from typing import Final, Protocol, cast
from unittest.mock import patch

import fakeredis
import pytest
from redis.asyncio.client import Pipeline
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm import Router
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.router_global_usage import GlobalUsageCounter, TransportRouterGlobalUsageStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache

TPM: Final = "global_router:deployment:openai/test-雪:tpm:12-34"
RPM: Final = "global_router:deployment:openai/test-雪:rpm:12-34"


class Clock(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return datetime(2026, 1, 1, 12, 34, 0, tzinfo=timezone.utc).astimezone(tz)


@pytest.fixture(autouse=True)
def clock() -> Iterator[None]:
    with (
        patch("litellm.utils.datetime", SimpleNamespace(datetime=Clock, timezone=timezone)),
        patch("time.time", return_value=1767270840.0),
    ):
        yield


class LegacyUsageCache(Protocol):
    async def async_set_cache(self, key: str, value: object, *, ttl: int) -> bool | None: ...

    async def async_get_cache(self, key: str) -> object: ...


class ExecutablePipeline(Protocol):
    async def execute(self, raise_on_error: bool = True) -> object: ...


class UsageCallbacks(Protocol):
    async def deployment_callback_on_success(
        self, kwargs: Mapping[str, object], completion_response: object, start_time: object, end_time: object
    ) -> object: ...

    async def async_deployment_callback_on_failure(
        self, kwargs: Mapping[str, object], completion_response: object, start_time: object, end_time: object
    ) -> object: ...


def _router(cache: RedisCache | None) -> Router:
    router: Final = Router(
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {"model": "openai/test-雪", "api_key": "fake"},
                "model_info": {"id": "deployment"},
                "tpm": 1000,
                "rpm": 100,
            }
        ]
    )
    if cache is not None:
        router.health().store.attach_redis_cache(cache)
    return router


async def _record(router: Router, tokens: float = 3, *, failure: bool = False) -> None:
    callbacks: Final = cast(UsageCallbacks, router)  # cast-ok: typed view of the legacy callback signatures
    metadata: Final = {"deployment": "openai/test-雪", "model_group": "group"}
    kwargs: Final[Mapping[str, object]] = {
        "call_type": "acompletion",
        "standard_logging_object": {"total_tokens": tokens},
        "litellm_params": {"metadata": metadata, "model_info": {"id": "deployment"}},
    }
    if failure:
        await callbacks.async_deployment_callback_on_failure(kwargs, None, None, None)
    else:
        await callbacks.deployment_callback_on_success(kwargs, None, None, None)


@pytest.mark.asyncio
async def test_usage_old_new_bytes_namespace_and_post_result_reconciliation(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    legacy: Final = cast(LegacyUsageCache, cache)  # cast-ok: legacy writer/reader for rolling-deploy probes
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await legacy.async_set_cache(TPM, 12, ttl=60)
    await legacy.async_set_cache(RPM, 2, ttl=60)
    router: Final = _router(cache)
    assert await router.get_model_group_usage("group") == (12, 2)
    with request_redis_batch_scope():
        await _record(router, 3.5)
    assert await raw.mget("tenant:" + TPM, "tenant:" + RPM) == [b"15.5", b"3"]
    assert await raw.exists(TPM, RPM) == 0
    assert await raw.pttl("tenant:" + TPM) == 60000
    assert await raw.pttl("tenant:" + RPM) == 60000
    assert await legacy.async_get_cache(TPM) == 15.5
    assert await legacy.async_get_cache(RPM) == 3
    assert await router.get_model_group_usage("group") == (15, 3)
    assert await _router(cache).get_model_group_usage("group") == (None, 3)


@pytest.mark.asyncio
async def test_failure_preserves_window_but_success_refreshes_it(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    router: Final = _router(cache)
    await _record(router, failure=True)
    assert await raw.get(RPM) == b"1"
    assert await raw.pttl(RPM) == 60000
    with patch("time.time", return_value=1767270850.0):
        await _record(router, failure=True)
        assert await raw.get(RPM) == b"2"
        assert await raw.pttl(RPM) == 50000
        await _record(router)
        assert await raw.get(RPM) == b"3"
        assert await raw.pttl(RPM) == 60000


@pytest.mark.asyncio
async def test_missing_usage_is_throttled_and_late_attachment_does_not_backfill(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    router: Final = _router(None)
    await _record(router)
    router.health().store.attach_redis_cache(cache)
    assert await raw.exists(TPM, RPM) == 0
    reader: Final = _router(cache)
    assert await reader.get_model_group_usage("group") == (None, None)
    await _record(router)
    assert await raw.mget(TPM, RPM) == [b"3", b"1"]
    assert await reader.get_model_group_usage("group") == (None, None)
    with patch("time.time", return_value=1767270860.0):
        assert await reader.get_model_group_usage("group") == (3, 1)


@pytest.mark.asyncio
async def test_malformed_batch_caches_raw_values_until_local_expiry(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(TPM, b"bad-json")
    await raw.set(RPM, b"5")
    router: Final = _router(cache)
    assert await router.get_model_group_usage("group") == (None, None)
    await raw.set(TPM, b"10")
    assert await router.get_model_group_usage("group") == (None, 5)
    assert await _router(cache).get_model_group_usage("group") == (10, 5)


@pytest.mark.asyncio
async def test_outage_keeps_local_counts_and_shared_breaker() -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server)
    router: Final = _router(cache)
    server.connected = False
    await _record(router)
    assert await router.get_model_group_usage("group") == (3, 1)
    store: Final = TransportRouterGlobalUsageStore(cache.transport())
    for _ in range(5):
        await store.read_global_usage((GlobalUsageCounter("deployment", "openai/test-雪", "12-34", "tpm"),))
    result: Final = await store.record_global_failure(
        "deployment", "openai/test-雪", "12-34", ttl=timedelta(seconds=60)
    )
    assert isinstance(result, RedisFailure) and result.kind == "breaker_open"
    server.connected = True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_unknown_applied_usage_write_is_not_replayed(server: fakeredis.FakeServer, failure: bool) -> None:
    cache: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    router: Final = _router(cache)
    original_execute: Final = cast(type[ExecutablePipeline], Pipeline).execute  # cast-ok: typed redis-py execution seam

    async def apply_then_disconnect(pipe: ExecutablePipeline, raise_on_error: bool = True) -> object:
        await original_execute(pipe, raise_on_error=raise_on_error)
        raise RedisConnectionError("response lost after pipeline applied")

    with patch.object(Pipeline, "execute", new=apply_then_disconnect):
        await _record(router, failure=failure)
    assert await raw.mget(TPM, RPM) == [None if failure else b"3", b"1"]
    assert await raw.pttl(RPM) == (-1 if failure else 60000)
    assert await router.get_model_group_usage("group") == (None if failure else 3, 1)
