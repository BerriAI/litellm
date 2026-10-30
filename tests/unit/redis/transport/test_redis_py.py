import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Final, Literal
from unittest.mock import AsyncMock, patch

import fakeredis
import pytest
from prometheus_client import REGISTRY
from redis.asyncio.client import Pipeline, Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

import litellm
from litellm._internal_context import current_service_target, service_target
from litellm._service_logger import ServiceLogging
from litellm.caching.redis_batch import (
    active_request_redis_batch,
    flush_post_call_redis_batches,
    request_redis_batch_scope,
)
from litellm.caching.redis_cluster_cache import ClusterConnectionGate, RedisClusterCache
from litellm.redis.transport.batch import PipelineBatch
from litellm.redis.transport.contract import (
    BatchReply,
    PubSubMessage,
    RedisFailure,
    RedisTransport,
    raise_redis_failure,
)
from litellm.redis.transport.redis_py import RedisPyCommandBatch, RedisPyTransport
from litellm.types.services import ServiceTypes
from tests._support.fake_redis import fake_redis_cache, fake_redis_transport


@dataclass
class _LazyRead:
    transport: RedisTransport
    reply: BatchReply[tuple[bytes | None, ...] | RedisFailure] | None = None

    def declare(self) -> None:
        batch: Final = self.transport.request_batch()
        if batch is not None and self.reply is None:
            self.reply = batch.mget(("present", "absent"))


@pytest.mark.asyncio
async def test_transport_flush_hook_joins_a_legacy_request_read(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="ns")
    transport: Final[RedisTransport] = cache.transport()
    await raw.set("ns:present", "3")
    with request_redis_batch_scope():
        batch: Final = transport.request_batch()
        legacy: Final = active_request_redis_batch(cache)
        assert batch is not None and legacy is not None
        reader: Final = _LazyRead(transport)
        batch.add_flush_hook(reader.declare)
        assert await legacy.mget(("present",)) == {"present": 3}
        assert reader.reply is not None
        assert await reader.reply == (b"3", None)
        assert legacy.flushes == 1
        assert batch.read_as_missing("absent") is True
        assert batch.read_as_missing("present") is False
        assert await transport.set("absent", "new", ttl=None) is True
        assert await batch.set("absent", "new", ttl=None) is True
        assert batch.read_as_missing("absent") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("outage", (False, True))
async def test_post_call_settled_hook_can_read_every_reply_and_rollback(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, outage: bool
) -> None:
    transport: Final[RedisTransport] = fake_redis_cache(server, namespace="ns").transport()
    with request_redis_batch_scope():
        batch: Final = transport.post_call_batch()
        assert batch is not None
        increment: Final = batch.incrbyfloat("counter", 2.5)
        expired: Final = batch.expire("counter", timedelta(seconds=30))
        observed: Final[asyncio.Future[tuple[float | RedisFailure, bool | RedisFailure]]] = (
            asyncio.get_running_loop().create_future()
        )

        async def settle(future: asyncio.Future[float | RedisFailure]) -> None:
            assert expired.done
            observed.set_result((future.result(), await expired))
            server.connected = True
            await transport.delete("counter")

        increment.on_settled(settle)
        server.connected = not outage
        await flush_post_call_redis_batches()
        first, second = await observed
        if outage:
            assert isinstance(first, RedisFailure) and first.kind == "transport"
            assert isinstance(second, RedisFailure) and second.kind == "transport"
        else:
            assert (first, second) == (2.5, True)
        assert await raw.get("ns:counter") is None


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
@pytest.mark.parametrize("mode", ("direct", "batch", "transaction"))
async def test_direct_eval_preserves_wire_replies_namespace_and_single_event(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, mode: str
) -> None:
    events: Final = _RecordingEvents()
    cache: Final = fake_redis_cache(server, namespace="ns", service_logger=events)
    await _settle()
    before: Final = len(events.seen)
    transport: Final = cache.transport()
    executor: Final = transport if mode == "direct" else transport.batch(transaction=mode == "transaction")

    assert await executor.eval_direct(_ECHO, ("ns:key",), ("value", 7)) == (b"value", 7, (b"ns:key",))
    assert await raw.get("ns:key") == b"value"
    assert await raw.get("ns:ns:key") is None
    await _settle()
    assert tuple((event.outcome, event.call_type) for event in events.seen[before:]) == (
        ("success", "eval_direct" if mode == "direct" else "redis_batch"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("direct", "batch", "transaction"))
async def test_direct_eval_never_replays_a_write_followed_by_noscript_error(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, mode: str
) -> None:
    events: Final = _RecordingEvents()
    cache: Final = fake_redis_cache(server, namespace="ns", service_logger=events)
    await _settle()
    before: Final = len(events.seen)
    transport: Final = cache.transport()
    executor: Final = transport if mode == "direct" else transport.batch(transaction=mode == "transaction")
    script: Final = "redis.call('INCR', KEYS[1]); return redis.error_reply('NOSCRIPT written already')"

    failure: Final = await executor.eval_direct(script, ("key",), ())

    assert isinstance(failure, RedisFailure) and failure.kind == "server"
    assert "written already" in failure.message
    assert await raw.get("ns:key") == b"1"
    await _settle()
    assert tuple((event.outcome, event.call_type) for event in events.seen[before:]) == (
        ("failure", "eval_direct") if mode == "direct" else ("success", "redis_batch"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("decode_responses", (False, True))
async def test_eval_preserves_empty_arrays_and_sibling_order(
    server: fakeredis.FakeServer, decode_responses: bool
) -> None:
    raw = fakeredis.FakeAsyncRedis(server=server, decode_responses=decode_responses)
    transport = fake_redis_transport(raw, failure_threshold=1)

    assert await transport.eval('return {{}, {1, {"hello", {}}}, {}, 0}', (), ()) == (
        (),
        (1, (b"hello", ())),
        (),
        0,
    )


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
        self.failures: tuple[str | Exception, ...] = ()

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
        self.failures = (*self.failures, error)


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
async def test_failure_events_keep_original_exception_objects(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    events: Final = _RecordingEvents()
    transport: Final = fake_redis_transport(raw, failure_threshold=1, events=events)
    await raw.set("wrong", "text")

    failed_command: Final = await transport.rpush("wrong", "item")
    server.connected = False
    failed_connection: Final = await transport.get("wrong")
    refused: Final = await transport.get("wrong")

    assert isinstance(failed_command, RedisFailure)
    assert isinstance(failed_connection, RedisFailure)
    assert isinstance(refused, RedisFailure) and refused.kind == "breaker_open"
    assert events.failures == (failed_command.exception, failed_connection.exception)
    assert isinstance(events.failures[0], ResponseError)
    assert isinstance(events.failures[1], RedisConnectionError)


@pytest.mark.asyncio
async def test_failed_request_metrics_preserve_redis_exception_classes(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "service_callback", ["prometheus_system"])
    labels: Final = {"redis": "redis", "error_class": "ResponseError", "function_name": "rpush"}
    connection_labels: Final = {"redis": "redis", "error_class": "ConnectionError", "function_name": "get"}
    transport: Final = fake_redis_transport(raw, failure_threshold=1, events=ServiceLogging())
    before: Final = REGISTRY.get_sample_value("litellm_redis_failed_requests_total", labels) or 0
    before_connection: Final = REGISTRY.get_sample_value("litellm_redis_failed_requests_total", connection_labels) or 0
    await raw.set("wrong", "text")

    failed_command: Final = await transport.rpush("wrong", "item")
    server.connected = False
    failed_connection: Final = await transport.get("wrong")
    refused: Final = await transport.get("wrong")

    assert isinstance(failed_command, RedisFailure) and failed_command.kind == "server"
    assert isinstance(failed_connection, RedisFailure) and failed_connection.kind == "transport"
    assert isinstance(refused, RedisFailure) and refused.kind == "breaker_open"
    assert REGISTRY.get_sample_value("litellm_redis_failed_requests_total", labels) == before + 1
    assert REGISTRY.get_sample_value("litellm_redis_failed_requests_total", connection_labels) == before_connection + 1


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
async def test_without_a_service_callback_no_event_is_built(
    server: fakeredis.FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "service_callback", [])
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


_Command = Callable[[RedisPyTransport], Awaitable[object]]


_WRONG_TYPE_COMMANDS: Final[tuple[_Command, ...]] = (
    lambda t: t.incrby("k", 1),
    lambda t: t.incrbyfloat("k", 1.0),
    lambda t: t.rpush("k", "x"),
    lambda t: t.sadd("k", "x"),
)
_OUTAGE_COMMANDS: Final[tuple[_Command, ...]] = (
    lambda t: t.incrby("k", 1),
    lambda t: t.expire("k", timedelta(seconds=1)),
    lambda t: t.expire_seconds("k", timedelta(seconds=1)),
    lambda t: t.set_seconds("k", "v", ttl=timedelta(seconds=1)),
    lambda t: t.ttl("k"),
    lambda t: t.lpop("k"),
    lambda t: t.scan("k", limit=1),
    lambda t: t.publish("c", "m"),
    lambda t: t.ping(),
)


@pytest.mark.asyncio
async def test_counters_add_to_the_stored_value_under_the_namespace(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()

    assert await transport.incrby("n", 2) == 2
    assert await transport.incrby("ns:n", 3) == 5
    assert await transport.incrbyfloat("f", 1.5) == 1.5
    assert await transport.incrbyfloat("f", 0.25) == 1.75

    assert await raw.get("ns:n") == b"5"
    assert await raw.get("ns:f") == b"1.75"


@pytest.mark.asyncio
async def test_expire_sets_the_ttl_in_milliseconds_and_ttl_reads_it_back(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()
    await raw.set("k", "v")

    assert await transport.ttl("k") == -1
    assert await transport.expire("k", timedelta(milliseconds=1500)) is True
    assert await transport.expire("absent", timedelta(seconds=5)) is False

    assert 1000 < await raw.pttl("k") <= 1500
    assert await transport.ttl("k") == 1
    assert await transport.ttl("absent") == -2


def _typed_pipeline(client: "Redis[bytes]") -> "Pipeline[bytes]":
    return client.pipeline(transaction=False)


def _seconds_commands(transport: RedisPyTransport, mode: str) -> RedisPyTransport | RedisPyCommandBatch:
    if mode == "direct":
        return transport
    if mode == "alone":
        backend: Final = replace(transport.batch().engine.backend, runs_alone=True)
        return RedisPyCommandBatch(transport, PipelineBatch(backend, name="seconds"))
    return transport.batch(transaction=mode == "transaction")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("direct", "batch", "transaction", "alone"))
@pytest.mark.parametrize("seconds", (2, 1.9, 0.5, 0, -0.5, -1))
async def test_native_expire_truncates_seconds_and_deletes_nonpositive_expiry(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, mode: str, seconds: float
) -> None:
    transport: Final = fake_redis_transport(raw, failure_threshold=1, namespace="ns")
    commands: Final = _seconds_commands(transport, mode)
    with patch("time.time", return_value=1700000000.0):
        await raw.set("ns:k", "v")
        assert await commands.expire_seconds("ns:k", timedelta(seconds=seconds)) is True
        assert await commands.expire_seconds("absent", timedelta(seconds=seconds)) is False
        if seconds >= 1:
            assert await raw.pttl("ns:k") == int(seconds) * 1000
            assert await raw.get("ns:k") == b"v"
        else:
            with patch("time.time", return_value=1700000000.001):
                assert await raw.get("ns:k") is None
        assert await raw.exists("ns:ns:k") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("direct", "batch", "transaction", "alone"))
async def test_native_set_preserves_seconds_expiry_persistence_and_nx(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, mode: str
) -> None:
    commands: Final = _seconds_commands(fake_redis_transport(raw, failure_threshold=1, namespace="ns"), mode)
    with patch("time.time", return_value=1700000000.0):
        assert await commands.set_seconds("k", b"1.25", ttl=timedelta(seconds=1.9), nx=True) is True
        assert await commands.set_seconds("ns:k", "rejected", ttl=None, nx=True) is False
        assert await raw.get("ns:k") == b"1.25"
        assert await raw.pttl("ns:k") == 1000
        assert await commands.set_seconds("ns:k", "persist", ttl=None) is True
        assert await raw.get("ns:k") == b"persist"
        assert await raw.pttl("ns:k") == -1
        assert await raw.keys("*") == [b"ns:k"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("direct", "batch", "transaction", "alone"))
@pytest.mark.parametrize("seconds", (0.5, 0, -0.5, -1))
async def test_native_set_rejection_keeps_value_and_connection_and_emits_one_event(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, mode: str, seconds: float
) -> None:
    events: Final = _RecordingEvents()
    transport: Final = fake_redis_transport(raw, failure_threshold=1, events=events, namespace="ns")
    commands: Final = _seconds_commands(transport, mode)
    await raw.set("ns:k", "before")
    with service_target("seconds_test"):
        rejected: Final = await commands.set_seconds("k", "after", ttl=timedelta(seconds=seconds))
    assert isinstance(rejected, RedisFailure) and rejected.kind == "server"
    assert await raw.get("ns:k") == b"before"
    assert len(events.seen) == 1
    assert events.seen[0].outcome == ("failure" if mode in ("direct", "alone") else "success")
    assert events.seen[0].target == "seconds_test"
    assert await commands.expire_seconds("k", timedelta(seconds=30)) is True
    await _settle()
    assert len(events.seen) == 2
    assert events.seen[1].outcome == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("direct", "batch"))
async def test_native_seconds_outage_opens_breaker_without_duplicate_events(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, mode: str
) -> None:
    events: Final = _RecordingEvents()
    commands: Final = _seconds_commands(fake_redis_transport(raw, failure_threshold=1, events=events), mode)
    server.connected = False
    failed: Final = await commands.expire_seconds("k", timedelta(seconds=30))
    assert isinstance(failed, RedisFailure) and failed.kind == "transport"
    server.connected = True
    refused: Final = await commands.set_seconds("k", "v", ttl=None)
    assert isinstance(refused, RedisFailure) and refused.kind == "breaker_open"
    await _settle()
    assert tuple(event.outcome for event in events.seen) == (
        ("failure",) if mode == "direct" else ("failure", "failure")
    )
    assert await raw.get("k") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("direct", "batch"))
@pytest.mark.parametrize("reply", (b"malformed", 2, 1.0, (True,), [True], {"unexpected": True}))
async def test_native_seconds_rejects_malformed_command_replies(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, mode: str, reply: object
) -> None:
    transport: Final = fake_redis_transport(raw, failure_threshold=1)
    pipeline: Final = _typed_pipeline(raw)
    with (
        patch.object(raw, "expire", new=AsyncMock(return_value=reply)),
        patch.object(raw, "set", new=AsyncMock(return_value=reply)),
        patch.object(raw, "pipeline", return_value=pipeline),
        patch.object(pipeline, "execute", return_value=[reply]),
    ):
        commands: Final = _seconds_commands(transport, mode)
        expired: Final = await commands.expire_seconds("k", timedelta(seconds=30))
        written: Final = await commands.set_seconds("k", "v", ttl=None)
    assert isinstance(expired, RedisFailure) and expired.kind == "decode"
    assert isinstance(written, RedisFailure) and written.kind == "decode"


@pytest.mark.asyncio
async def test_native_expire_lost_reply_is_unknown_applied_without_retry(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport: Final = fake_redis_transport(raw, failure_threshold=1)
    expire: Final = raw.expire
    await raw.set("k", "v")

    async def lose_reply(name: str, time: int) -> bool:
        await expire(name, time)
        raise RedisConnectionError("reply lost after expiry applied")

    with patch("time.time", return_value=1700000000.0), patch.object(raw, "expire", side_effect=lose_reply):
        result: Final = await transport.expire_seconds("k", timedelta(seconds=30))
        assert isinstance(result, RedisFailure) and result.kind == "transport"
        assert await raw.pttl("k") == 30000


@pytest.mark.asyncio
async def test_list_commands_push_trim_and_pop_from_the_head(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()

    assert await transport.rpush("l", "a", "b", "c", "d") == 4
    assert await transport.ltrim("l", -3, -1) is True
    assert await transport.lpop("l") == b"b"
    assert await transport.lpop("l", 5) == (b"c", b"d")
    assert await transport.lpop("l") is None
    assert await transport.lpop("l", 2) is None

    assert await raw.exists("ns:l") == 0


@pytest.mark.asyncio
async def test_lpop_with_a_count_before_redis_7_pops_the_same_items(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = replace(fake_redis_cache(server).transport(), server_version=lambda: 6)
    await raw.rpush("l", "a", "b", "c")

    assert await transport.lpop("l", 2) == (b"a", b"b")
    assert await transport.lpop("l", 5) == (b"c",)
    assert await transport.lpop("l", 5) is None


@pytest.mark.asyncio
async def test_sadd_counts_only_new_members(server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()

    assert await transport.sadd("s", "a", "b") == 2
    assert await transport.sadd("s", "b", "c") == 1

    assert await raw.smembers("ns:s") == {b"a", b"b", b"c"}


@pytest.mark.asyncio
async def test_scan_finds_keys_under_the_namespaced_prefix_up_to_the_limit(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()
    for key in ("ns:team:1", "ns:team:2", "ns:team:3", "ns:user:1", "team:4"):
        await raw.set(key, "v")

    assert sorted(await transport.scan("team:", limit=10)) == [b"ns:team:1", b"ns:team:2", b"ns:team:3"]
    assert len(await transport.scan("team:", limit=2)) == 2


@pytest.mark.parametrize("command", _WRONG_TYPE_COMMANDS)
@pytest.mark.asyncio
async def test_a_command_on_the_wrong_type_is_a_server_failure(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, command: _Command
) -> None:
    transport = fake_redis_cache(server).transport()
    await raw.hset("k", "f", "v")

    failure = await command(transport)

    assert isinstance(failure, RedisFailure) and failure.kind == "server"


@pytest.mark.parametrize("command", _OUTAGE_COMMANDS)
@pytest.mark.asyncio
async def test_an_outage_is_a_transport_failure_for_every_command(
    server: fakeredis.FakeServer, command: _Command
) -> None:
    transport = fake_redis_cache(server).transport()
    server.connected = False

    failure = await command(transport)

    assert isinstance(failure, RedisFailure) and failure.kind == "transport"


class _ClusterClient:
    def __init__(self, inner: fakeredis.FakeAsyncRedis) -> None:
        self.inner = inner

    async def mget(self, keys: Sequence[str]) -> object:
        raise ResponseError("CROSSSLOT Keys in request don't hash to the same slot")

    async def mget_nonatomic(self, keys: Sequence[str]) -> object:
        return [await self.inner.get(key) for key in keys]


@pytest.mark.asyncio
async def test_mget_on_a_cluster_cache_reads_keys_from_different_slots(raw: fakeredis.FakeAsyncRedis) -> None:
    with patch("litellm._redis.init_redis_cluster"):
        cache = RedisClusterCache(startup_nodes=[{"host": "localhost", "port": 7000}])
    litellm.in_memory_llm_clients_cache.set_cache(
        key=cache._get_async_client_cache_key(),  # pyright: ignore[reportPrivateUsage]  # seeds the per-loop client slot RedisCache reads
        value=_ClusterClient(raw),
    )
    await raw.set("a", "1")
    await raw.set("b", "2")

    assert await cache.transport().mget(("a", "b", "c")) == (b"1", b"2", None)


@pytest.mark.asyncio
async def test_a_batch_sends_every_declared_command_in_one_round_trip(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    events = _RecordingEvents()
    transport = fake_redis_cache(server, namespace="ns", service_logger=events).transport()
    await _settle()
    await raw.set("ns:a", "1")
    before = len(events.seen)
    batch = transport.batch()

    got = batch.get("a")
    written = batch.set("b", "2", ttl=timedelta(seconds=30))
    counted = batch.incrbyfloat("n", 1.5)
    pushed = batch.rpush("l", "x", "y")
    read = batch.mget(("a", "b", "missing"))
    assert not got.done

    assert await got == b"1"
    assert (await written, await counted, await pushed) == (True, 1.5, 2)
    assert await read == (b"1", b"2", None)
    await _settle()
    assert [event.call_type for event in events.seen[before:]] == ["redis_batch"]
    assert 0 < await raw.pttl("ns:b") <= 30_000


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", (False, True))
async def test_explicit_batch_error_policy_preserves_other_writes_and_event_outcome(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, strict: bool
) -> None:
    events: Final = _RecordingEvents()
    cache: Final = fake_redis_cache(server, namespace="ns", service_logger=events)
    await _settle()
    await raw.set("ns:wrong", "text")
    before: Final = len(events.seen)
    batch: Final = cache.transport().batch(raise_on_error=strict, name="legacy_batch")
    bad: Final = batch.rpush("wrong", "item")
    good: Final = batch.set("written", "value", ttl=None)

    failure: Final = await bad
    assert isinstance(failure, RedisFailure) and failure.kind == "server"
    with pytest.raises(ResponseError, match="WRONGTYPE") as caught:
        raise_redis_failure(failure)
    assert caught.value is failure.exception
    assert await good == (failure if strict else True)
    assert await raw.get("ns:written") == b"value"
    assert await raw.get("ns:wrong") == b"text"
    await _settle()
    assert tuple((event.outcome, event.call_type) for event in events.seen[before:]) == (
        ("failure" if strict else "success", "legacy_batch"),
    )


@pytest.mark.asyncio
async def test_transport_failure_retains_original_exception_for_public_facade(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport: Final = fake_redis_transport(raw, failure_threshold=1)
    server.connected = False

    failure: Final = await transport.get("key")
    assert isinstance(failure, RedisFailure) and failure.kind == "transport"
    with pytest.raises(RedisConnectionError) as caught:
        raise_redis_failure(failure)
    assert caught.value is failure.exception
    assert str(caught.value) == failure.message
    assert failure == RedisFailure(kind="transport", message=failure.message)
    assert "exception=" not in repr(failure)


@pytest.mark.asyncio
async def test_a_failing_command_in_a_batch_fails_only_its_own_reply(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()
    await raw.hset("h", "f", "v")
    batch = transport.batch()

    bad = batch.incrby("h", 1)
    good = batch.incrby("n", 1)

    failure = await bad
    assert isinstance(failure, RedisFailure) and failure.kind == "server"
    assert await good == 1


@pytest.mark.asyncio
async def test_an_outage_fails_every_reply_in_the_batch_and_counts_once_on_the_breaker(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_transport(raw, failure_threshold=1)
    batch = transport.batch()
    server.connected = False

    declared = (batch.get("a"), batch.incrby("n", 1))
    replies = [await reply for reply in declared]
    server.connected = True
    refused = await transport.get("a")

    assert all(isinstance(reply, RedisFailure) and reply.kind == "transport" for reply in replies)
    assert isinstance(refused, RedisFailure) and refused.kind == "breaker_open"


@pytest.mark.asyncio
async def test_a_script_redis_does_not_have_cached_still_runs_in_a_batch(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()
    await raw.script_flush()
    batch = transport.batch()

    scripted = batch.eval("return redis.call('INCRBY', KEYS[1], ARGV[1])", ("n",), (4,))
    counted = batch.incrby("other", 1)

    assert await scripted == 4
    assert await counted == 1
    assert await raw.get("n") == b"4"


@pytest.mark.asyncio
async def test_a_transaction_batch_runs_its_commands_atomically_with_uncached_scripts(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server).transport()
    await raw.script_flush()
    batch = transport.batch(transaction=True)

    pushed = batch.rpush("l", "a", "b", "c")
    trimmed = batch.ltrim("l", -2, -1)
    scripted = batch.eval("return redis.call('LLEN', KEYS[1])", ("l",), ())

    assert (await pushed, await trimmed, await scripted) == (3, True, 2)
    assert await raw.lrange("l", 0, -1) == [b"b", b"c"]


@pytest.mark.asyncio
async def test_a_transaction_stays_one_round_trip_on_a_backend_that_runs_batched_commands_alone(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    events = _RecordingEvents()
    transport = fake_redis_transport(raw, failure_threshold=1, events=events)
    alone_backend = replace(transport.batch().engine.backend, runs_alone=True)
    batch = RedisPyCommandBatch(transport, PipelineBatch(alone_backend, name="tx", transaction=True))

    pushed = batch.rpush("l", "a", "b")
    trimmed = batch.ltrim("l", -1, -1)

    assert (await pushed, await trimmed) == (2, True)
    await _settle()
    assert [event.call_type for event in events.seen] == ["tx"]


@pytest.mark.asyncio
async def test_request_batch_commands_share_the_round_trip_of_redis_batch_callers(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache = fake_redis_cache(server, namespace="ns")
    await raw.set("ns:legacy", '"old"')
    await raw.set("ns:moved", "new")
    transport = cache.transport()
    assert transport.request_batch() is None

    with request_redis_batch_scope():
        legacy = active_request_redis_batch(cache)
        commands = transport.request_batch()
        assert legacy is not None and commands is not None
        old_read = legacy.mget(["legacy"])
        new_read = commands.get("moved")
        missing = commands.get("absent")
        written = commands.set_seconds("seconds", "v", ttl=None)
        expired = commands.expire_seconds("seconds", timedelta(seconds=30))

        assert await old_read == {"legacy": "old"}
        assert await new_read == b"new"
        assert await missing is None
        assert await written is True and await expired is True
        assert legacy.flushes == 1
        assert legacy.read_as_missing("absent") is True
        assert legacy.read_as_missing("moved") is False


@pytest.mark.asyncio
async def test_publish_reaches_a_subscriber_on_the_namespaced_channel(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_cache(server, namespace="ns").transport()
    subscription = await transport.subscribe("config")
    assert not isinstance(subscription, RedisFailure)

    assert await transport.publish("config", "one") == 1
    assert await transport.publish_all("config", ("two", "three")) == (1, 1)
    assert await raw.publish("config", "elsewhere") == 0

    received = [await subscription.get_message(timeout=1) for _ in range(4)]
    messages = [message for message in received if message is not None]
    assert messages == [
        PubSubMessage(channel=b"ns:config", data=b"one"),
        PubSubMessage(channel=b"ns:config", data=b"two"),
        PubSubMessage(channel=b"ns:config", data=b"three"),
    ]
    assert await subscription.close() is None


@pytest.mark.asyncio
async def test_health_and_admin_commands_still_run_while_the_breaker_is_open(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    transport = fake_redis_transport(raw, failure_threshold=1)
    await raw.set("k", "v")
    server.connected = False
    await transport.get("k")
    server.connected = True
    assert isinstance(refused := await transport.get("k"), RedisFailure) and refused.kind == "breaker_open"

    assert await transport.ping() is True
    clients = await transport.client_list()
    assert not isinstance(clients, RedisFailure) and len(clients) >= 1
    assert await transport.flushall() is True
    assert await raw.exists("k") == 0


class _InfoClient:
    async def info(self) -> object:
        return {"redis_version": "7.4.0", "connected_clients": 3, "db0": {"keys": 2, "expires": 0}}


@pytest.mark.asyncio
async def test_info_returns_the_parsed_sections(raw: fakeredis.FakeAsyncRedis) -> None:
    transport = replace(fake_redis_transport(raw, failure_threshold=1), client=lambda: _InfoClient())

    info = await transport.info()

    assert not isinstance(info, RedisFailure)
    assert (info["redis_version"], info["connected_clients"], info["db0"]) == ("7.4.0", 3, {"keys": 2, "expires": 0})
