import logging
import time
from datetime import timedelta
from typing import Final
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm._internal_context import current_service_target
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.file_usage_caps import (
    FileUsageLimit,
    FileUsageLimitExceeded,
    ScopedFileUsageLimit,
    consume_file_usage,
    enforce_batch_file_upload_limit,
    enforce_file_download_limit,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.file_usage_counters import (
    CounterScope,
    FileUsageCounter,
    FileUsageSetting,
    TransportFileUsageCounterStore,
)
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache

NOW: Final = 1_800_000_000
DOWNLOAD: Final = "max_file_downloads_per_minute"
UPLOAD: Final = "max_batch_file_uploads_per_day"


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: float(NOW))


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["key", "user", "team"])
@pytest.mark.parametrize("setting", ["max_batch_file_records", UPLOAD, DOWNLOAD])
async def test_old_writer_new_caller_and_new_writer_old_reader_preserve_numeric_bytes_and_namespace(
    scope: CounterScope, setting: FileUsageSetting, server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    redis: Final = fake_redis_cache(server, namespace="ns")
    cache: Final = InternalUsageCache(dual_cache=DualCache(redis_cache=redis))
    key: Final = f"litellm:file_usage:{setting}:{scope}:identity:file:é:{NOW}"
    limits: Final = (ScopedFileUsageLimit(scope, "identity", FileUsageLimit(setting, 4, "key")),)
    assert await redis.async_increment(key, 1.25, ttl=60) == 1.25

    assert await consume_file_usage(cache, limits, 60, "file:é", NOW + 15) is None
    assert await raw.get(f"ns:{key}") == b"2.25"
    assert await redis.async_get_cache(key) == 2.25
    assert await redis.async_increment(key, 0.5, ttl=60) == 2.75
    assert await raw.get(f"ns:{key}") == b"2.75"
    assert await raw.pttl(f"ns:{key}") == 60_000
    assert await raw.keys("*") == [f"ns:{key}".encode()]
    assert await cache.dual_cache.async_get_cache(key, local_only=True) == 1


@pytest.mark.asyncio
async def test_increment_preserves_existing_expiry_and_repairs_an_unexpiring_counter(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server)
    cache: Final = InternalUsageCache(dual_cache=DualCache(redis_cache=redis))
    limits: Final = (ScopedFileUsageLimit("key", "identity", FileUsageLimit(DOWNLOAD, 10, "key")),)
    key: Final = f"litellm:file_usage:{DOWNLOAD}:key:identity:file:{NOW}"

    assert await consume_file_usage(cache, limits, 60, "file", NOW) is None
    assert await raw.get(key) == b"1"
    assert await raw.pttl(key) == 60_000
    monkeypatch.setattr(time, "time", lambda: float(NOW + 7))
    assert await consume_file_usage(cache, limits, 60, "file", NOW + 7) is None
    assert await raw.get(key) == b"2"
    assert await raw.pttl(key) == 53_000
    assert cache.dual_cache.in_memory_cache.ttl_dict[key] == NOW + 60
    await raw.persist(key)
    assert await consume_file_usage(cache, limits, 60, "file", NOW + 7) is None
    assert await raw.get(key) == b"3"
    assert await raw.pttl(key) == 60_000


@pytest.mark.asyncio
async def test_later_scope_rejection_compensates_visited_scopes_only(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    redis: Final = fake_redis_cache(server, namespace="ns")
    cache: Final = InternalUsageCache(dual_cache=DualCache(redis_cache=redis))
    limits: Final = (
        ScopedFileUsageLimit("key", "key", FileUsageLimit(DOWNLOAD, 10, "key")),
        ScopedFileUsageLimit("team", "team", FileUsageLimit(DOWNLOAD, 1, "team")),
        ScopedFileUsageLimit("user", "user", FileUsageLimit(DOWNLOAD, 10, "key")),
    )
    team_key: Final = f"ns:litellm:file_usage:{DOWNLOAD}:team:team:file:{NOW}"
    key_key: Final = f"ns:litellm:file_usage:{DOWNLOAD}:key:key:file:{NOW}"
    await raw.set(team_key, "1", px=22_000)

    assert await consume_file_usage(cache, limits, 60, "file", NOW + 15) == FileUsageLimitExceeded(limits[1], 45)
    assert await raw.get(key_key) == b"0"
    assert await raw.get(team_key) == b"1"
    assert await raw.pttl(key_key) == 60_000
    assert await raw.pttl(team_key) == 22_000
    assert set(await raw.keys("*")) == {key_key.encode(), team_key.encode()}
    assert await cache.dual_cache.async_get_cache(key_key.removeprefix("ns:"), local_only=True) == 0
    assert await cache.dual_cache.async_get_cache(team_key.removeprefix("ns:"), local_only=True) == 0


@pytest.mark.asyncio
async def test_outage_and_breaker_keep_bounded_local_admission(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, caplog: pytest.LogCaptureFixture
) -> None:
    redis: Final = fake_redis_cache(server, namespace="ns")
    cache: Final = InternalUsageCache(dual_cache=DualCache(redis_cache=redis))
    limit: Final = ScopedFileUsageLimit("key", "identity", FileUsageLimit(DOWNLOAD, 1, "key"))
    server.connected = False

    with caplog.at_level(logging.WARNING):
        assert await consume_file_usage(cache, (limit,), 60, "file", NOW) is None
        for _ in range(10):
            assert await consume_file_usage(cache, (limit,), 60, "file", NOW + 15) == FileUsageLimitExceeded(limit, 45)
    assert "pipeline of 2 ops failed" in caplog.text
    assert current_service_target() is None
    server.connected = True
    failure: Final = await TransportFileUsageCounterStore(redis.transport()).increment(
        FileUsageCounter(DOWNLOAD, "key", "identity", "file", NOW), 1, ttl=timedelta(seconds=60)
    )
    assert isinstance(failure, RedisFailure)
    assert failure.kind == "breaker_open"
    assert await raw.keys("*") == []
    assert await cache.dual_cache.async_get_cache(
        f"litellm:file_usage:{DOWNLOAD}:key:identity:file:{NOW}", local_only=True
    ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_type", [False, True])
async def test_server_errors_use_local_count_without_replacing_malformed_redis_values(
    wrong_type: bool, server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    redis: Final = fake_redis_cache(server)
    cache: Final = InternalUsageCache(dual_cache=DualCache(redis_cache=redis))
    key: Final = f"litellm:file_usage:{DOWNLOAD}:key:identity:file:{NOW}"
    if wrong_type:
        await raw.rpush(key, "not-a-counter")
    else:
        await raw.set(key, "not-a-counter")
    counter: Final = FileUsageCounter(DOWNLOAD, "key", "identity", "file", NOW)
    failure: Final = await TransportFileUsageCounterStore(redis.transport()).increment(
        counter, 1, ttl=timedelta(seconds=60)
    )
    assert isinstance(failure, RedisFailure)
    assert failure.kind == "server"
    limit: Final = ScopedFileUsageLimit("key", "identity", FileUsageLimit(DOWNLOAD, 1, "key"))

    assert await consume_file_usage(cache, (limit,), 60, "file", NOW) is None
    assert await consume_file_usage(cache, (limit,), 60, "file", NOW + 15) == FileUsageLimitExceeded(limit, 45)
    assert await raw.pttl(key) == -1
    assert (await raw.lrange(key, 0, -1) if wrong_type else await raw.get(key)) == (
        [b"not-a-counter"] if wrong_type else b"not-a-counter"
    )
    assert await redis.transport().get("healthy") is None


@pytest.mark.asyncio
async def test_corrupt_local_counter_still_prevents_redis_increment(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = InternalUsageCache(dual_cache=DualCache(redis_cache=fake_redis_cache(server)))
    key: Final = f"litellm:file_usage:{DOWNLOAD}:key:identity:file:{NOW}"
    await cache.dual_cache.async_set_cache(key, "not-a-counter", local_only=True)
    limit: Final = ScopedFileUsageLimit("key", "identity", FileUsageLimit(DOWNLOAD, 1, "key"))

    assert await consume_file_usage(cache, (limit,), 60, "file", NOW) is None
    assert await raw.get(key) is None
    assert await cache.dual_cache.async_get_cache(key, local_only=True) == "not-a-counter"


@pytest.mark.asyncio
async def test_local_only_fallback_preserves_integer_precision_above_float_range() -> None:
    cache: Final = InternalUsageCache(dual_cache=DualCache())
    key: Final = f"litellm:file_usage:{DOWNLOAD}:key:identity:file:{NOW}"
    maximum: Final = 2**53
    await cache.dual_cache.async_set_cache(key, maximum, local_only=True, ttl=60)
    limit: Final = ScopedFileUsageLimit("key", "identity", FileUsageLimit(DOWNLOAD, maximum, "key"))

    assert await consume_file_usage(cache, (limit,), 60, "file", NOW + 15) == FileUsageLimitExceeded(limit, 45)
    assert await cache.dual_cache.async_get_cache(key, local_only=True) == maximum


@pytest.mark.asyncio
async def test_applied_increment_with_lost_expiry_reply_is_not_replayed(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, caplog: pytest.LogCaptureFixture
) -> None:
    redis: Final = fake_redis_cache(server, namespace="ns")
    cache: Final = InternalUsageCache(dual_cache=DualCache(redis_cache=redis))
    key: Final = f"ns:litellm:file_usage:{DOWNLOAD}:key:identity:file:{NOW}"
    limit: Final = ScopedFileUsageLimit("key", "identity", FileUsageLimit(DOWNLOAD, 5, "key"))
    client: Final = redis.init_async_client()

    async def applied_then_timeout(*args: object, **kwargs: object) -> None:
        await raw.pexpire(key, 60_000)
        raise RedisTimeoutError("expiry applied but reply lost")

    with patch.object(client, "pexpire", side_effect=applied_then_timeout) as expire:
        assert await consume_file_usage(cache, (limit,), 60, "file", NOW) is None
        assert expire.call_count == 1
    assert "rate_limits" in caplog.text
    assert current_service_target() is None
    assert await raw.get(key) == b"1"
    assert await raw.pttl(key) == 60_000
    assert await cache.dual_cache.async_get_cache(key.removeprefix("ns:"), local_only=True) == 1
    assert await consume_file_usage(cache, (limit,), 60, "file", NOW + 15) is None
    assert await raw.get(key) == b"2"


@pytest.mark.asyncio
async def test_increment_pipeline_with_lost_reply_is_not_replayed_or_given_an_expiry(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    redis: Final = fake_redis_cache(server)
    cache: Final = InternalUsageCache(dual_cache=DualCache(redis_cache=redis))
    key: Final = f"litellm:file_usage:{DOWNLOAD}:key:identity:file:{NOW}"
    limit: Final = ScopedFileUsageLimit("key", "identity", FileUsageLimit(DOWNLOAD, 5, "key"))
    client: Final = redis.init_async_client()
    pipeline: Final = client.pipeline(transaction=False)
    execute: Final = pipeline.execute

    async def applied_then_timeout(*args: object, **kwargs: object) -> None:
        await execute(*args, **kwargs)
        raise RedisTimeoutError("increment applied but reply lost")

    with (
        patch.object(client, "pipeline", return_value=pipeline),
        patch.object(pipeline, "execute", side_effect=applied_then_timeout) as execution,
    ):
        assert await consume_file_usage(cache, (limit,), 60, "file", NOW) is None
        assert execution.call_count == 1
    assert await raw.get(key) == b"1"
    assert await raw.pttl(key) == -1
    assert await cache.dual_cache.async_get_cache(key, local_only=True) == 1
    assert await consume_file_usage(cache, (limit,), 60, "file", NOW + 15) is None
    assert await raw.get(key) == b"2"
    assert await raw.pttl(key) == 60_000


@pytest.mark.asyncio
async def test_upload_and_download_callers_keep_scope_subject_windows_and_error_contract(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = InternalUsageCache(dual_cache=DualCache(redis_cache=fake_redis_cache(server)))
    caller: Final = UserAPIKeyAuth(api_key="key", user_id="user", team_id="team")
    limits: Final = {UPLOAD: 1, DOWNLOAD: 1}
    day_start: Final = NOW // 86400 * 86400

    await enforce_batch_file_upload_limit(cache, caller, limits, clock=lambda: NOW)
    with pytest.raises(ProxyException) as upload:
        await enforce_batch_file_upload_limit(cache, caller, limits, clock=lambda: NOW + 15)
    assert upload.value.code == "429"
    assert upload.value.type == "rate_limit_error"
    assert upload.value.headers == {"retry-after": str(day_start + 86400 - NOW - 15)}
    assert await raw.get(f"litellm:file_usage:{UPLOAD}:key:key::{day_start}") == b"1"
    assert await raw.pttl(f"litellm:file_usage:{UPLOAD}:key:key::{day_start}") == 86_400_000
    await enforce_file_download_limit(cache, caller, limits, "file-a", clock=lambda: NOW)
    with pytest.raises(ProxyException) as download:
        await enforce_file_download_limit(cache, caller, limits, "file-a", clock=lambda: NOW + 15)
    assert download.value.code == "429"
    assert download.value.type == "rate_limit_error"
    assert download.value.headers == {"retry-after": "45"}
    assert f"{DOWNLOAD} is 1 for this key" in download.value.message
    await enforce_file_download_limit(cache, caller, limits, "file-b", clock=lambda: NOW)
    await enforce_file_download_limit(cache, caller, limits, "file-a", clock=lambda: NOW + 60)
    assert await raw.get(f"litellm:file_usage:{DOWNLOAD}:key:key:file-a:{NOW}") == b"1"
    assert await raw.get(f"litellm:file_usage:{DOWNLOAD}:key:key:file-b:{NOW}") == b"1"
    assert await raw.get(f"litellm:file_usage:{DOWNLOAD}:key:key:file-a:{NOW + 60}") == b"1"
