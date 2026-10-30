import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Literal

from unittest.mock import patch

import fakeredis
import pytest

import litellm

from litellm._internal_context import current_service_target, service_target
from litellm._service_logger import ServiceLogging
from litellm.caching.redis_cluster_cache import ClusterConnectionGate, RedisClusterCache
from litellm.redis.transport.contract import RedisFailure
from litellm.types.services import ServiceTypes
from tests._support.fake_redis import fake_redis_cache, fake_redis_transport


@pytest.mark.asyncio
async def test_every_command_prefixes_the_namespace_once(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()

    assert await transport.set("k", "v", ttl=None) is True
    assert await transport.set("ns:gone", "v", ttl=None) is True

    assert sorted(await raw.keys("*")) == [b"ns:gone", b"ns:k"]
    assert await transport.get("k") == await transport.get("ns:k") == b"v"
    assert await transport.delete("gone") == 1
    assert await raw.keys("*") == [b"ns:k"]


@pytest.mark.asyncio
async def test_set_nx_refuses_a_live_key_and_applies_the_ttl(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()

    assert await transport.set("k", "first", ttl=timedelta(seconds=30), nx=True) is True
    assert await transport.set("k", "second", ttl=timedelta(seconds=30), nx=True) is False

    assert await raw.get("k") == b"first"
    assert 0 < await raw.pttl("k") <= 30_000


@pytest.mark.asyncio
async def test_set_without_ttl_persists(server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis) -> None:
    transport = fake_redis_cache(server).transport()

    assert await transport.set("k", "v", ttl=None) is True

    assert await raw.pttl("k") == -1


@pytest.mark.asyncio
async def test_delete_reports_how_many_keys_it_removed(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()
    await raw.set("k", "v")

    assert await transport.delete("k") == 1
    assert await transport.delete("k") == 0
    assert await raw.exists("k") == 0


@pytest.mark.asyncio
async def test_a_rejected_command_is_a_server_failure_and_keeps_the_connection(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()
    await raw.rpush("list", "x")

    rejected = await transport.get("list")

    assert isinstance(rejected, RedisFailure)
    assert rejected.kind == "server"
    assert await transport.set("k", "v", ttl=None) is True


@pytest.mark.asyncio
async def test_an_unreachable_server_is_a_transport_failure(server: fakeredis.FakeServer) -> None:
    transport = fake_redis_cache(server).transport()
    server.connected = False

    failure = await transport.get("k")

    assert isinstance(failure, RedisFailure)
    assert failure.kind == "transport"


@pytest.mark.asyncio
async def test_a_transport_failure_opens_the_breaker_and_it_answers_without_redis(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_transport(raw, failure_threshold=1)
    server.connected = False

    tripped = await transport.get("k")
    server.connected = True
    refused = await transport.get("k")

    assert isinstance(tripped, RedisFailure)
    assert tripped.kind == "transport"
    assert isinstance(refused, RedisFailure)
    assert refused.kind == "breaker_open"


@pytest.mark.asyncio
async def test_mget_returns_one_slot_per_key_in_order(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()
    await raw.set("ns:a", "1")
    await raw.set("ns:c", "3")

    assert await transport.mget(("a", "b", "ns:c")) == (b"1", None, b"3")
    assert await transport.mget(()) == ()


@pytest.mark.asyncio
async def test_delete_takes_several_keys(server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()
    await raw.set("ns:a", "1")
    await raw.set("ns:b", "2")
    await raw.set("c", "3")

    assert await transport.delete("a", "b", "c") == 2
    assert await raw.keys("*") == [b"c"]


_ECHO: Final = """
redis.call("set", KEYS[1], ARGV[1])
if ARGV[2] == "false" then
    return false
end
return {redis.call("get", KEYS[1]), tonumber(ARGV[2]), {KEYS[1]}}
"""


@pytest.mark.asyncio
async def test_eval_namespaces_keys_and_returns_nested_replies(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()

    assert await transport.eval(_ECHO, ("k",), ("v", 7)) == (b"v", 7, (b"ns:k",))
    assert await transport.eval(_ECHO, ("k",), ("w", "false")) is None
    assert await raw.get("ns:k") == b"w"


@pytest.mark.asyncio
async def test_eval_survives_a_script_flush(server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis) -> None:
    transport = fake_redis_cache(server).transport()

    assert await transport.eval(_ECHO, ("k",), ("v", 1)) == (b"v", 1, (b"k",))
    await raw.script_flush()
    assert await transport.eval(_ECHO, ("k",), ("w", 2)) == (b"w", 2, (b"k",))


@pytest.mark.asyncio
async def test_a_script_error_is_a_server_failure(server: fakeredis.FakeServer) -> None:
    transport = fake_redis_cache(server).transport()

    failure = await transport.eval('return redis.error_reply("nope")', (), ())

    assert isinstance(failure, RedisFailure)
    assert failure.kind == "server"
    assert await transport.eval("return 1", (), ()) == 1


class _OneAtATimeClient:
    def __init__(self, inner: fakeredis.FakeAsyncRedis) -> None:
        self.inner = inner
        self.in_flight = 0
        self.peak_in_flight = 0

    async def get(self, name: str) -> object:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return await self.inner.get(name)


@pytest.mark.asyncio
async def test_a_cluster_cache_transport_waits_on_the_connection_gate(raw: fakeredis.FakeAsyncRedis) -> None:
    with patch("litellm._redis.init_redis_cluster"):
        cache = RedisClusterCache(
            startup_nodes=[{"host": "localhost", "port": 7000}],
            connection_gate=ClusterConnectionGate(limit=1, wait_timeout=5.0),
        )
    client = _OneAtATimeClient(raw)
    litellm.in_memory_llm_clients_cache.set_cache(
        key=cache._get_async_client_cache_key(),  # pyright: ignore[reportPrivateUsage]  # seeds the per-loop client slot RedisCache reads
        value=client,
    )
    await raw.set("k", "v")
    transport = cache.transport()

    replies = await asyncio.gather(*(transport.get("k") for _ in range(5)))

    assert replies == [b"v"] * 5
    assert client.peak_in_flight == 1


@pytest.mark.asyncio
async def test_replies_stay_bytes_on_a_client_that_decodes_responses(server: fakeredis.FakeServer) -> None:
    transport = fake_redis_transport(
        fakeredis.FakeAsyncRedis(server=server, decode_responses=True), failure_threshold=1
    )

    assert await transport.set("k", "v", ttl=None) is True
    assert await transport.get("k") == b"v"


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

    def has_sinks(self) -> bool:
        return True

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
        await transport.set("k", "v", ttl=None)
        await transport.get("k")
        server.connected = False
        failed = await transport.delete("k")

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

    await transport.get("k")
    refused = await transport.get("k")

    assert isinstance(refused, RedisFailure) and refused.kind == "breaker_open"
    assert tuple(event.outcome for event in events.seen) == ("failure",)


@pytest.mark.asyncio
async def test_the_cache_transport_reports_to_the_caches_service_logger(server: fakeredis.FakeServer) -> None:
    events = _RecordingEvents()
    transport = fake_redis_cache(server, service_logger=events).transport()
    await _settle()
    before = events.seen

    await transport.get("k")
    server.connected = False
    await transport.get("k")

    assert tuple(event.outcome for event in events.seen[len(before) :]) == ("success", "failure")


@pytest.mark.asyncio
async def test_without_a_service_callback_no_event_is_built(server: fakeredis.FakeServer) -> None:
    service_logger = ServiceLogging(mock_testing=True)
    transport = fake_redis_cache(server, service_logger=service_logger).transport()
    await _settle()
    before = (service_logger.mock_testing_async_success_hook, service_logger.mock_testing_async_failure_hook)

    assert await transport.get("k") is None
    server.connected = False
    assert isinstance(await transport.get("k"), RedisFailure)
    await _settle()

    after = (service_logger.mock_testing_async_success_hook, service_logger.mock_testing_async_failure_hook)
    assert after == before


@pytest.mark.asyncio
async def test_a_failure_is_logged_once_with_the_command_and_target(
    server: fakeredis.FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    transport = fake_redis_cache(server).transport()
    await _settle()
    server.connected = False
    caplog.clear()

    with caplog.at_level(logging.WARNING), service_target("response_polling"):
        failure = await transport.get("k")

    assert isinstance(failure, RedisFailure)
    assert [record.getMessage() for record in caplog.records] == [
        f"Redis get failed for response_polling: {failure.message}"
    ]
