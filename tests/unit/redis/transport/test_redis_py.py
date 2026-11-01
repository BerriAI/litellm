import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Literal

import fakeredis
import pytest

from litellm._internal_context import current_service_target, service_target
from litellm._service_logger import ServiceLogging
from litellm.redis.transport.contract import Delete, Get, RedisFailure, Set
from litellm.types.services import ServiceTypes
from tests._support.fake_redis import fake_redis_cache, fake_redis_transport


@pytest.mark.asyncio
async def test_every_command_prefixes_the_namespace_once(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()

    assert await transport.execute(Set("k", "v", ttl=None)) is True
    assert await transport.execute(Set("ns:gone", "v", ttl=None)) is True

    assert sorted(await raw.keys("*")) == [b"ns:gone", b"ns:k"]
    assert await transport.execute(Get("k")) == await transport.execute(Get("ns:k")) == b"v"
    assert await transport.execute(Delete("gone")) == 1
    assert await raw.keys("*") == [b"ns:k"]


@pytest.mark.asyncio
async def test_set_nx_refuses_a_live_key_and_applies_the_ttl(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()

    assert await transport.execute(Set("k", "first", ttl=timedelta(seconds=30), nx=True)) is True
    assert await transport.execute(Set("k", "second", ttl=timedelta(seconds=30), nx=True)) is False

    assert await raw.get("k") == b"first"
    assert 0 < await raw.pttl("k") <= 30_000


@pytest.mark.asyncio
async def test_set_without_ttl_persists(server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis) -> None:
    transport = fake_redis_cache(server).transport()

    assert await transport.execute(Set("k", "v", ttl=None)) is True

    assert await raw.pttl("k") == -1


@pytest.mark.asyncio
async def test_delete_reports_how_many_keys_it_removed(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()
    await raw.set("k", "v")

    assert await transport.execute(Delete("k")) == 1
    assert await transport.execute(Delete("k")) == 0
    assert await raw.exists("k") == 0


@pytest.mark.asyncio
async def test_a_rejected_command_is_a_server_failure_and_keeps_the_connection(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()
    await raw.rpush("list", "x")

    rejected = await transport.execute(Get("list"))

    assert isinstance(rejected, RedisFailure)
    assert rejected.kind == "server"
    assert await transport.execute(Set("k", "v", ttl=None)) is True


@pytest.mark.asyncio
async def test_an_unreachable_server_is_a_transport_failure(server: fakeredis.FakeServer) -> None:
    transport = fake_redis_cache(server).transport()
    server.connected = False

    failure = await transport.execute(Get("k"))

    assert isinstance(failure, RedisFailure)
    assert failure.kind == "transport"


@pytest.mark.asyncio
async def test_a_transport_failure_opens_the_breaker_and_it_answers_without_redis(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_transport(raw, failure_threshold=1)
    server.connected = False

    tripped = await transport.execute(Get("k"))
    server.connected = True
    refused = await transport.execute(Get("k"))

    assert isinstance(tripped, RedisFailure)
    assert tripped.kind == "transport"
    assert isinstance(refused, RedisFailure)
    assert refused.kind == "breaker_open"


@pytest.mark.asyncio
async def test_replies_stay_bytes_on_a_client_that_decodes_responses(server: fakeredis.FakeServer) -> None:
    transport = fake_redis_transport(
        fakeredis.FakeAsyncRedis(server=server, decode_responses=True), failure_threshold=1
    )

    assert await transport.execute(Set("k", "v", ttl=None)) is True
    assert await transport.execute(Get("k")) == b"v"


async def _settle() -> None:
    pending: Final = {task for task in asyncio.all_tasks() if task is not asyncio.current_task()}
    if pending:
        await asyncio.gather(*pending)
        await _settle()


@dataclass(frozen=True, slots=True)
class _Event:
    outcome: Literal["success", "failure"]
    call_type: str
    target: str | None
    failed_with: str | None = None


class _RecordingEvents(ServiceLogging):
    def __init__(self) -> None:
        super().__init__()
        self.seen: tuple[_Event, ...] = ()

    async def async_service_success_hook(
        self,
        service: ServiceTypes,
        call_type: str,
        duration: float,
        parent_otel_span: object | None = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: object | None = None,
        caller: str | None = None,
    ) -> None:
        assert service == ServiceTypes.REDIS and 0 <= duration
        self.seen = (*self.seen, _Event("success", call_type, current_service_target()))

    async def async_service_failure_hook(
        self,
        service: ServiceTypes,
        duration: float,
        error: str | Exception,
        call_type: str,
        parent_otel_span: object | None = None,
        start_time: datetime | float | None = None,
        end_time: float | datetime | None = None,
        event_metadata: object | None = None,
        caller: str | None = None,
    ) -> None:
        assert service == ServiceTypes.REDIS and 0 <= duration
        self.seen = (*self.seen, _Event("failure", call_type, current_service_target(), str(error)))


@pytest.mark.asyncio
async def test_every_command_reports_one_service_event_under_the_callers_target(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    events = _RecordingEvents()
    transport = fake_redis_transport(raw, failure_threshold=2, events=events)

    with service_target("response_polling"):
        await transport.execute(Set("k", "v", ttl=None))
        await transport.execute(Get("k"))
        server.connected = False
        failed = await transport.execute(Delete("k"))
    await asyncio.sleep(0)

    assert isinstance(failed, RedisFailure)
    assert events.seen == (
        _Event("success", "set", "response_polling"),
        _Event("success", "get", "response_polling"),
        _Event("failure", "delete", "response_polling", failed.message),
    )


@pytest.mark.asyncio
async def test_a_call_the_breaker_refuses_reports_no_service_event(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    events = _RecordingEvents()
    transport = fake_redis_transport(raw, failure_threshold=1, events=events)
    server.connected = False

    await transport.execute(Get("k"))
    refused = await transport.execute(Get("k"))
    await asyncio.sleep(0)

    assert isinstance(refused, RedisFailure) and refused.kind == "breaker_open"
    assert tuple(event.outcome for event in events.seen) == ("failure",)


@pytest.mark.asyncio
async def test_the_cache_transport_reports_to_the_caches_service_logger(server: fakeredis.FakeServer) -> None:
    service_logger = ServiceLogging(mock_testing=True)
    transport = fake_redis_cache(server, service_logger=service_logger).transport()
    await _settle()
    before = (service_logger.mock_testing_async_success_hook, service_logger.mock_testing_async_failure_hook)

    await transport.execute(Get("k"))
    server.connected = False
    await transport.execute(Get("k"))
    await _settle()

    after = (service_logger.mock_testing_async_success_hook, service_logger.mock_testing_async_failure_hook)
    assert (after[0] - before[0], after[1] - before[1]) == (1, 1)


@pytest.mark.asyncio
async def test_a_failure_is_logged_once_with_the_command_and_target(
    server: fakeredis.FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    transport = fake_redis_cache(server).transport()
    await _settle()
    server.connected = False
    caplog.clear()

    with caplog.at_level(logging.WARNING), service_target("response_polling"):
        failure = await transport.execute(Get("k"))

    assert isinstance(failure, RedisFailure)
    assert [record.getMessage() for record in caplog.records] == [
        f"Redis get failed for response_polling: {failure.message}"
    ]
