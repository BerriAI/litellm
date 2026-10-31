import asyncio
import json
import logging
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import ResponseError

from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.redis.stores.pod_lock import HeldBy
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache

LOCK_KEY = "cronjob_lock:test_job"


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def raw(server: fakeredis.FakeServer) -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=server)


async def _settle() -> None:
    pending = {task for task in asyncio.all_tasks() if task is not asyncio.current_task()}
    if pending:
        await asyncio.gather(*pending)
        await _settle()


async def _legacy_acquire(cache: RedisCache, pod_id: str, ttl: int = DEFAULT_CRON_JOB_LOCK_TTL_SECONDS) -> object:
    return await cache.async_set_cache(PodLockManager.get_redis_lock_key("test_job"), pod_id, nx=True, ttl=ttl)


@pytest.mark.asyncio
async def test_acquire_writes_the_json_encoded_pod_id_with_the_ttl(server: fakeredis.FakeServer, raw) -> None:
    cache = fake_redis_cache(server)
    pod = PodLockManager(redis_cache=cache)

    assert await pod.acquire_lock(cronjob_id="test_job", ttl=120) is True

    assert await raw.get(LOCK_KEY) == json.dumps(pod.pod_id).encode()
    assert 119_000 < await raw.pttl(LOCK_KEY) <= 120_000
    assert await cache.async_get_cache(LOCK_KEY) == pod.pod_id


@pytest.mark.asyncio
async def test_acquire_defaults_to_the_cron_job_ttl(server: fakeredis.FakeServer, raw) -> None:
    cache = fake_redis_cache(server)
    assert await PodLockManager(redis_cache=cache).acquire_lock(cronjob_id="test_job") is True

    assert (
        (DEFAULT_CRON_JOB_LOCK_TTL_SECONDS - 1) * 1000
        < await raw.pttl(LOCK_KEY)
        <= DEFAULT_CRON_JOB_LOCK_TTL_SECONDS * 1000
    )


@pytest.mark.asyncio
async def test_a_lock_taken_by_an_old_pod_is_refused_to_others_and_reentrant_for_it(
    server: fakeredis.FakeServer,
) -> None:
    cache = fake_redis_cache(server)
    holder = PodLockManager(redis_cache=cache)
    other = PodLockManager(redis_cache=cache)
    await _legacy_acquire(cache, holder.pod_id)

    assert await other.acquire_lock(cronjob_id="test_job") is False
    assert await holder.acquire_lock(cronjob_id="test_job") is True
    assert await holder.acquire_lock(cronjob_id="test_job", allow_reentrant=False) is False
    assert await holder.lock_holder("test_job") == HeldBy(holder.pod_id)
    assert await other.is_held("test_job") is True


@pytest.mark.asyncio
async def test_only_one_of_many_concurrent_pods_acquires(server: fakeredis.FakeServer) -> None:
    cache = fake_redis_cache(server)
    pods = [PodLockManager(redis_cache=cache) for _ in range(5)]

    results = await asyncio.gather(*(pod.acquire_lock(cronjob_id="test_job") for pod in pods))

    assert sorted(results) == [False, False, False, False, True]


@pytest.mark.asyncio
async def test_release_deletes_only_the_owners_lock(server: fakeredis.FakeServer, raw) -> None:
    cache = fake_redis_cache(server)
    holder = PodLockManager(redis_cache=cache)
    other = PodLockManager(redis_cache=cache)
    assert await holder.acquire_lock(cronjob_id="test_job") is True

    await other.release_lock(cronjob_id="test_job")
    assert await raw.get(LOCK_KEY) == json.dumps(holder.pod_id).encode()

    await holder.release_lock(cronjob_id="test_job")
    assert await raw.exists(LOCK_KEY) == 0
    assert await other.acquire_lock(cronjob_id="test_job") is True


@pytest.mark.asyncio
async def test_release_frees_a_lock_written_by_an_old_pod(server: fakeredis.FakeServer, raw) -> None:
    cache = fake_redis_cache(server)
    pod = PodLockManager(redis_cache=cache)
    await _legacy_acquire(cache, pod.pod_id)

    await pod.release_lock(cronjob_id="test_job")

    assert await raw.exists(LOCK_KEY) == 0


@pytest.mark.asyncio
async def test_release_survives_a_script_flush(server: fakeredis.FakeServer, raw) -> None:
    cache = fake_redis_cache(server)
    pod = PodLockManager(redis_cache=cache)
    assert await pod.acquire_lock(cronjob_id="test_job") is True
    await pod.release_lock(cronjob_id="test_job")
    assert await pod.acquire_lock(cronjob_id="test_job") is True
    await raw.script_flush()

    await pod.release_lock(cronjob_id="test_job")

    assert await raw.exists(LOCK_KEY) == 0


class _ScriptingDisabled(fakeredis.FakeAsyncRedis):
    async def eval(self, *args: object) -> object:
        raise ResponseError("ERR scripting is disabled")

    async def evalsha(self, *args: object) -> object:
        raise ResponseError("ERR scripting is disabled")


@pytest.mark.asyncio
async def test_release_falls_back_to_get_and_delete_when_scripts_cannot_run(server: fakeredis.FakeServer, raw) -> None:
    cache = fake_redis_cache(server, client=_ScriptingDisabled)
    holder = PodLockManager(redis_cache=cache)
    other = PodLockManager(redis_cache=cache)
    assert await holder.acquire_lock(cronjob_id="test_job") is True

    await other.release_lock(cronjob_id="test_job")
    assert await raw.get(LOCK_KEY) == json.dumps(holder.pod_id).encode()

    with patch.object(holder, "_emit_released_lock_event") as emitted:
        await holder.release_lock(cronjob_id="test_job")
    assert await raw.exists(LOCK_KEY) == 0
    emitted.assert_called_once_with(cronjob_id="test_job", pod_id=holder.pod_id)


@pytest.mark.asyncio
async def test_release_emits_the_released_event_only_when_it_deleted_the_lock(server: fakeredis.FakeServer) -> None:
    cache = fake_redis_cache(server)
    pod = PodLockManager(redis_cache=cache)

    with patch.object(pod, "_emit_released_lock_event") as emitted:
        await pod.release_lock(cronjob_id="test_job")
        assert emitted.call_count == 0
        await pod.acquire_lock(cronjob_id="test_job")
        await pod.release_lock(cronjob_id="test_job")

    emitted.assert_called_once_with(cronjob_id="test_job", pod_id=pod.pod_id)


@pytest.mark.asyncio
async def test_an_owner_that_does_not_decode_is_a_failure_and_not_acquired(server: fakeredis.FakeServer, raw) -> None:
    cache = fake_redis_cache(server)
    await raw.set(LOCK_KEY, b"not json")
    pod = PodLockManager(redis_cache=cache)

    assert await pod.acquire_lock(cronjob_id="test_job") is False
    holder = await pod.lock_holder("test_job")
    assert isinstance(holder, RedisFailure) and holder.kind == "decode"


@pytest.mark.asyncio
async def test_an_unreadable_lock_is_reported_unheld_with_a_warning_naming_the_job(
    server: fakeredis.FakeServer, raw, caplog: pytest.LogCaptureFixture
) -> None:
    await raw.set(LOCK_KEY, b"not json")
    pod = PodLockManager(redis_cache=fake_redis_cache(server))

    with caplog.at_level(logging.WARNING):
        assert await pod.is_held("test_job") is False

    assert any("test_job" in record.getMessage() for record in caplog.records if record.levelno == logging.WARNING)


@pytest.mark.asyncio
async def test_an_unreachable_redis_is_not_acquired_and_logs_no_error(
    server: fakeredis.FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    pod = PodLockManager(redis_cache=fake_redis_cache(server))
    await _settle()
    server.connected = False
    caplog.clear()

    with caplog.at_level(logging.ERROR):
        assert await pod.acquire_lock(cronjob_id="test_job") is False
        await pod.release_lock(cronjob_id="test_job")
    holder = await pod.lock_holder("test_job")

    assert isinstance(holder, RedisFailure) and holder.kind == "transport"
    assert [record for record in caplog.records if record.levelno >= logging.ERROR] == []


@pytest.mark.asyncio
async def test_without_redis_nothing_is_acquired_or_held() -> None:
    pod = PodLockManager(redis_cache=None)

    assert await pod.acquire_lock(cronjob_id="test_job") is None
    await pod.release_lock(cronjob_id="test_job")
    assert await pod.lock_holder("test_job") is None
    assert await pod.is_held("test_job") is False


@pytest.mark.asyncio
async def test_a_namespaced_cache_keeps_the_lock_under_its_namespace(server: fakeredis.FakeServer, raw) -> None:
    pod = PodLockManager(redis_cache=fake_redis_cache(server, namespace="ns"))

    assert await pod.acquire_lock(cronjob_id="test_job") is True

    assert await raw.keys("*") == [f"ns:{LOCK_KEY}".encode()]
